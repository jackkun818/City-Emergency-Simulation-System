import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from src.core import config
import json
from src.core.environment import Environment
from gymnasium import spaces

class RescuerAgent(nn.Module):
    """单个救援人员的智能体模型"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, device=None):
        super(RescuerAgent, self).__init__()
        # 设置设备
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值网络头 - 评估状态价值
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 策略网络头 - 产生动作概率
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        # 确保输入在正确的设备上
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        elif x.device != self.device:
            x = x.to(self.device)
            
        features = self.feature_layer(x)
        value = self.value_head(features)
        policy_logits = self.policy_head(features)
        return value, policy_logits
    
    def get_action(self, state, epsilon=0.0):
        """使用ε-贪心策略选择动作"""
        if random.random() < epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.policy_head.out_features - 1)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                value, policy_logits = self.forward(state)
                return torch.argmax(policy_logits).item()


class ExperienceReplay:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class MARLController:
    """多智能体强化学习控制器，管理所有救援人员智能体"""
    def __init__(self, grid_size=None, num_rescuers=None, hidden_dim=128, lr=0.001, gamma=0.99):
        """初始化MARL控制器"""
        self.grid_size = grid_size or config.get_config_param("grid_size")
        self.num_rescuers = num_rescuers or config.get_config_param("num_rescuers")
        if isinstance(num_rescuers, list):  # 如果传入的是rescuers列表
            self.num_rescuers = len(num_rescuers)
        
        self.gamma = gamma
        
        # 设置设备
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 状态空间：
        # 1. 救援人员位置 (x, y)
        # 2. 救援人员专长类型
        # 3. 救援人员效能值
        # 4. 每个灾情点的类型和严重程度
        # 5. 每个灾情点的当前救援资源分配情况
        # 状态维度计算: 救援人员自身状态(4) + 灾情信息(grid_size*grid_size*5)
        self.state_dim = 4 + self.grid_size * self.grid_size * 5
        
        # 动作空间：
        # 0: 不动
        # 1~grid_size*grid_size: 前往对应格子坐标的灾情点
        self.action_dim = 1 + self.grid_size * self.grid_size
        
        # 为每个救援人员创建一个智能体
        self.agents = [RescuerAgent(self.state_dim, self.action_dim, hidden_dim, device=self._device) for _ in range(self.num_rescuers)]
        self.optimizers = [optim.Adam(agent.parameters(), lr=lr) for agent in self.agents]
        
        # 经验回放缓冲区
        self.replay_buffers = [ExperienceReplay(config.MARL_CONFIG["replay_buffer_size"]) for _ in range(self.num_rescuers)]
        
        # 目标网络
        self.target_agents = [RescuerAgent(self.state_dim, self.action_dim, hidden_dim, device=self._device) for _ in range(self.num_rescuers)]
        for i in range(self.num_rescuers):
            self.target_agents[i].load_state_dict(self.agents[i].state_dict())

        # 更新计数器，用于定期更新目标网络
        self.update_counter = 0
        
        # 训练参数
        self.epsilon = config.EPSILON_START if hasattr(config, "EPSILON_START") else config.MARL_CONFIG["epsilon_start"]
        self.epsilon_start = config.EPSILON_START if hasattr(config, "EPSILON_START") else config.MARL_CONFIG["epsilon_start"]
        self.epsilon_end = config.EPSILON_FINAL if hasattr(config, "EPSILON_FINAL") else config.MARL_CONFIG["epsilon_end"]
        self.epsilon_decay = config.MARL_CONFIG["epsilon_decay"]
        self.batch_size = config.MARL_CONFIG["batch_size"]
        self.target_update = config.MARL_CONFIG["target_update"]
        self.steps_done = 0
        
        # 环境缓存，用于协调
        self._env_cache = []
        
        # 奖励跟踪
        self.reward_tracking = {
            "completion_reward": [],  # 完成任务奖励
            "priority_reward": [],    # 高优先级任务奖励
            "coordination_reward": [], # 协调奖励
            "progress_reward": [],    # 救援进度奖励
            "time_penalty": []        # 时间惩罚
        }
        
        # 目录不存在则创建
        os.makedirs(os.path.dirname(config.MARL_CONFIG["model_save_path"]), exist_ok=True)
    
    @property
    def device(self):
        """获取当前设备"""
        return self._device
    
    def build_state(self, rescuer_idx, rescuers, disasters):
        """构建单个智能体的状态表示"""
        if rescuer_idx >= len(rescuers):
            print(f"警告: rescuer_idx ({rescuer_idx}) 超出了 rescuers 列表长度 ({len(rescuers)})")
            # 返回零向量作为应急措施
            return torch.zeros((1, self.state_dim), device=self.device)
            
        rescuer = rescuers[rescuer_idx]
        
        # 提取救援人员自身状态
        state = [
            rescuer["position"][0] / self.grid_size,  # 标准化x坐标
            rescuer["position"][1] / self.grid_size,  # 标准化y坐标
            1.0 if "target" in rescuer and rescuer["target"] is not None else 0.0,  # 是否已分配任务
            rescuer.get("speed", config.MAX_SPEED) / config.MAX_SPEED  # 标准化移动速度
        ]
        
        # 网格状态矩阵初始化
        grid_state = np.zeros((self.grid_size, self.grid_size, 5))
        
        # 填充灾情信息
        for pos, disaster in disasters.items():
            x, y = pos
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # 灾情存在标志
                grid_state[x, y, 0] = 1.0  
                # 标准化灾情等级
                grid_state[x, y, 1] = disaster["level"] / config.CRITICAL_DISASTER_THRESHOLD  
                # 标准化所需救援资源
                grid_state[x, y, 2] = disaster["rescue_needed"] / config.MAX_RESCUE_CAPACITY  
                # 标准化已分配资源
                assigned_count = sum(1 for r in rescuers if "target" in r and r["target"] == pos)
                grid_state[x, y, 3] = assigned_count / self.num_rescuers  
                # 剩余所需救援资源
                grid_state[x, y, 4] = disaster["rescue_needed"] / config.MAX_RESCUE_CAPACITY
        
        # 将网格状态展平并添加到状态向量
        grid_state_flat = grid_state.flatten()
        state.extend(grid_state_flat)
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 添加批处理维度并移至正确设备
    
    def select_action(self, state, rescuer_idx, disasters=None):
        """为单个救援人员选择动作，使用动态动作掩码"""
        # 检查rescuer_idx是否超出范围
        if rescuer_idx >= self.num_rescuers:
            print(f"警告: rescuer_idx ({rescuer_idx}) 超出了范围 ({self.num_rescuers})")
            return 0  # 返回不动作
            
        # 确保state是张量且在正确设备上
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        # 创建动作掩码 - 初始化所有动作为无效
        action_mask = torch.zeros(self.action_dim, device=self.device)
        
        # 如果提供了灾情信息，创建动态掩码
        if disasters is not None and len(disasters) > 0:
            # 对每个灾情点，将对应的动作标记为有效
            for pos in disasters.keys():
                x, y = pos
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    action_idx = x * self.grid_size + y + 1  # +1 是因为动作0是"不动"
                    if 0 < action_idx < self.action_dim:  # 确保动作索引有效
                        action_mask[action_idx] = 1.0
            
            # 如果没有有效动作（没有灾情点），则不动
            if action_mask.sum() == 0:
                return 0
        else:
            # 如果没有提供灾情信息，允许所有动作（除了不动）
            action_mask[1:] = 1.0
            
        # 使用epsilon-greedy策略选择动作
        if random.random() < self.epsilon:
            # 探索：从有效动作中随机选择
            valid_actions = torch.nonzero(action_mask).squeeze(-1)
            if len(valid_actions) > 0:
                # 从有效动作中随机选择
                random_idx = random.randint(0, len(valid_actions) - 1)
                return valid_actions[random_idx].item()
            else:
                # 如果没有有效动作，则不动
                return 0
        else:
            # 利用：选择Q值最大的有效动作
            with torch.no_grad():
                _, policy_logits = self.agents[rescuer_idx](state)
                
                # 应用掩码：将无效动作的logits设为负无穷
                masked_logits = policy_logits.clone()
                masked_logits[0, action_mask == 0] = float('-inf')
                
                # 如果所有动作都被掩码，则返回不动作
                if (masked_logits[0] == float('-inf')).all():
                    return 0
                
                # 选择最大Q值的有效动作
                return torch.argmax(masked_logits).item()
    
    def select_actions(self, rescuers, disasters, training=False):
        """为所有救援人员选择动作，兼容旧版本API"""
        actions = []
        
        # 更新探索率
        if training:
            self.steps_done += 1
        
        for i in range(self.num_rescuers):
            state = self.build_state(i, rescuers, disasters)
            # 使用select_action方法选择动作，传递灾情信息
            action = self.select_action(state, i, disasters)
            actions.append(action)
        
        return actions
    
    def store_transition(self, state, action, reward, next_state, done, rescuer_idx):
        """存储单个智能体的经验"""
        # 检查rescuer_idx是否超出范围
        if rescuer_idx >= self.num_rescuers:
            print(f"警告: store_transition - rescuer_idx ({rescuer_idx}) 超出了范围 ({self.num_rescuers})")
            return
            
        # 确保数据类型正确
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        self.replay_buffers[rescuer_idx].push(state, action, reward, next_state, done)
    
    def store_experience(self, rescuer_idx, state, action, reward, next_state, done):
        """兼容旧版本API的经验存储方法"""
        self.store_transition(state, action, reward, next_state, done, rescuer_idx)
    
    def update_agent(self, agent_idx):
        """更新单个智能体"""
        if agent_idx >= self.num_rescuers:
            print(f"警告: update_agent - agent_idx ({agent_idx}) 超出了范围 ({self.num_rescuers})")
            return 0.0
            
        if len(self.replay_buffers[agent_idx]) < self.batch_size:
            return 0.0
        
        # 从经验回放中采样
        transitions = self.replay_buffers[agent_idx].sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # 准备批数据
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        # 确保state_batch形状正确 - 如果state是二维张量(batch_size, 1, state_dim)，则需要压缩一维
        if len(state_batch.shape) == 3 and state_batch.shape[1] == 1:
            state_batch = state_batch.squeeze(1)
            
        # 确保action_batch是长整型并且维度正确
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        # 检查并确保action_batch是二维的
        if len(action_batch.shape) == 1:
            action_batch = action_batch.unsqueeze(1)  # 添加第二维
        
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        
        # 检查奖励是否有异常值
        if torch.isnan(reward_batch).any() or torch.isinf(reward_batch).any():
            # 替换NaN或无限值为0
            reward_batch = torch.nan_to_num(reward_batch, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 同样处理next_state_batch
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        if len(next_state_batch.shape) == 3 and next_state_batch.shape[1] == 1:
            next_state_batch = next_state_batch.squeeze(1)
            
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)
        
        try:
            # 计算当前Q值
            current_value, policy_logits = self.agents[agent_idx](state_batch)
            
            # 确保actions在有效范围内
            action_batch = torch.clamp(action_batch, 0, self.action_dim - 1)
            
            # 现在gather操作应该能正确执行
            q_values = policy_logits.gather(1, action_batch)
            
            # 计算目标Q值
            with torch.no_grad():
                next_value, next_policy_logits = self.target_agents[agent_idx](next_state_batch)
                next_q_values = next_policy_logits.max(1, keepdim=True)[0]
                expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
            # 检查Q值是否有异常值
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                # 如果发现异常值，返回一个默认损失值，跳过此次更新
                print(f"警告：智能体 {agent_idx} 的Q值包含NaN或无限值，跳过更新")
                return 0.0
                
            # 检查目标Q值是否有异常值
            if torch.isnan(expected_q_values).any() or torch.isinf(expected_q_values).any():
                # 如果发现异常值，处理目标Q值
                expected_q_values = torch.nan_to_num(expected_q_values, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # 计算损失 - 使用合适的损失函数
            loss = F.smooth_l1_loss(q_values, expected_q_values)
            
            # 检查损失值是否异常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告：智能体 {agent_idx} 的损失值为 {loss.item()}，跳过更新")
                return 0.0
                
            # 防止梯度爆炸，设置损失上限
            loss_value = loss.item()
            if loss_value > 1000.0:
                # 缩放损失以避免极大值
                loss = loss * (1000.0 / loss_value)
                
            # 优化模型
            self.optimizers[agent_idx].zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].parameters(), max_norm=10.0)
                
            self.optimizers[agent_idx].step()
            
            return min(loss.item(), 1000.0)  # 返回有限的损失值
            
        except Exception as e:
            print(f"更新智能体 {agent_idx} 时出错: {e}")
            return 0.0
    
    def update_agents(self):
        """更新所有智能体"""
        total_loss = 0.0
        updated_count = 0
        
        for i in range(self.num_rescuers):
            loss = self.update_agent(i)
            if loss > 0:
                total_loss += loss
                updated_count += 1
            
            # 如果满足更新目标网络的条件，则更新目标网络
            if self.steps_done % self.target_update == 0:
                self.target_agents[i].load_state_dict(self.agents[i].state_dict())
        
        # 返回平均损失，用于监控训练进度
        return total_loss / max(1, updated_count) if updated_count > 0 else 0.0
    
    def track_rewards(self, reward_info):
        """跟踪不同类型的奖励"""
        for reward_type, value in reward_info.items():
            if reward_type in self.reward_tracking:
                self.reward_tracking[reward_type].append(value)
    
    def get_reward_stats(self, reset=True):
        """获取奖励统计信息并可选择性地重置跟踪数据"""
        stats = {}
        for reward_type, values in self.reward_tracking.items():
            if values:  # 如果有数据
                stats[reward_type] = {
                    "total": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
            else:
                stats[reward_type] = {"total": 0, "avg": 0, "min": 0, "max": 0}
                
        if reset:
            # 重置跟踪数据
            for reward_type in self.reward_tracking:
                self.reward_tracking[reward_type] = []
                
        return stats
    
    def save_models(self, path=None):
        """保存所有智能体模型"""
        if path is None:
            path = config.MARL_CONFIG["model_save_path"]
        
        model_states = {
            f"agent_{i}": self.agents[i].state_dict() for i in range(self.num_rescuers)
        }
        torch.save(model_states, path)
        print(f"模型已保存到 {path}")
    
    def load_models(self, path=None):
        """加载所有智能体模型"""
        if path is None:
            path = config.MARL_CONFIG["model_save_path"]
        
        if not os.path.exists(path):
            print(f"没有找到模型文件: {path}")
            return False
        
        try:
            model_states = torch.load(path)
            for i in range(self.num_rescuers):
                if f"agent_{i}" in model_states:
                    self.agents[i].load_state_dict(model_states[f"agent_{i}"])
                    self.target_agents[i].load_state_dict(model_states[f"agent_{i}"])
            print(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False

    def action_to_target(self, action, grid_size=None):
        """
        将动作索引转换为目标位置坐标
        
        参数:
            action: 动作索引（0表示不动，1~grid_size*grid_size表示目标位置）
            grid_size: 网格大小，如果为None则使用self.grid_size
            
        返回:
            (x, y): 目标位置坐标，如果action为0则返回None
        """
        if action <= 0:
            return None
            
        if grid_size is None:
            grid_size = self.grid_size
            
        action_idx = action - 1
        x = action_idx // grid_size
        y = action_idx % grid_size
        
        return (x, y)


class RescueEnvironment:
    def __init__(self, grid_size=None, num_rescuers=None, rescuers_data=None):
        """初始化救援环境"""
        self.env = Environment(grid_size=grid_size, num_rescuers=num_rescuers, rescuers_data=rescuers_data)
        # 设置动作空间
        self.action_space = spaces.Discrete(5)  # 4 方向移动 + 不移动
        # 设置观察空间 (多个特征平面: 灾情, 救援者位置等)
        grid_size = grid_size or config.get_config_param("grid_size")
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32)
        self.current_time_step = 0
        self.previous_success_count = 0
        self.disaster_count = 0

    def get_rescuers_data(self):
        """获取救援人员数据，用于保存"""
        return self.env.rescuers
    
    def save_rescuers_data(self, file_path):
        """保存救援人员数据到文件"""
        rescuers_data = self.get_rescuers_data()
        # 将rescuers转换为可序列化的字典列表
        serialized_rescuers = []
        for rescuer in rescuers_data:
            # 检查rescuer是否为字典类型
            if isinstance(rescuer, dict):
                # 如果已经是字典，直接复制
                rescuer_dict = rescuer.copy()
                # 确保position是列表格式（JSON不支持元组）
                if 'position' in rescuer_dict and isinstance(rescuer_dict['position'], tuple):
                    rescuer_dict['position'] = list(rescuer_dict['position'])
            else:
                # 如果是对象，提取属性
                rescuer_dict = {
                    'id': rescuer.id,
                    'position': list(rescuer.position) if hasattr(rescuer, 'position') else [0, 0],
                    'specialty': rescuer.specialty if hasattr(rescuer, 'specialty') else None,
                    'effectiveness': rescuer.effectiveness if hasattr(rescuer, 'effectiveness') else 1.0,
                    'capacity': rescuer.capacity if hasattr(rescuer, 'capacity') else 1,
                    'speed': rescuer.speed if hasattr(rescuer, 'speed') else 1,
                    'active_time': rescuer.active_time if hasattr(rescuer, 'active_time') else 0
                }
            serialized_rescuers.append(rescuer_dict)
        
        # 保存为JSON文件
        with open(file_path, 'w') as f:
            json.dump(serialized_rescuers, f, indent=4)
        
        return serialized_rescuers
        
    @staticmethod
    def load_rescuers_data(file_path):
        """从文件加载救援人员数据"""
        with open(file_path, 'r') as f:
            rescuers_data = json.load(f)
        
        # 将数据转换为适当的格式
        rescuers = []
        for rescuer_dict in rescuers_data:
            # 确保position是元组格式
            if 'position' in rescuer_dict and isinstance(rescuer_dict['position'], list):
                rescuer_dict['position'] = tuple(rescuer_dict['position'])
            rescuers.append(rescuer_dict)
        
        return rescuers

    def reset(self):
        """重置环境"""
        # 这个方法在现有的系统中不需要实现，因为环境已经在main.py中初始化
        self.current_time_step = 0
        return self.env
    
    def get_state_for_rescuer(self, rescuer_idx):
        """获取指定救援人员的状态"""
        if not hasattr(self, 'marl_controller') or self.marl_controller is None:
            # 如果没有MARL控制器实例，创建一个临时的
            from src.rl.marl_rescue import MARLController
            self.marl_controller = MARLController(
                grid_size=self.env.GRID_SIZE,
                num_rescuers=self.env.rescuers
            )
        
        # 使用MARL控制器的状态构建方法
        return self.marl_controller.build_state(rescuer_idx, self.env.rescuers, self.env.disasters)
    
    def is_episode_done(self):
        """检查当前回合是否结束"""
        # 检查是否所有灾难都已解决
        all_resolved = True
        for disaster in self.env.disasters.values():
            if disaster["rescue_needed"] > 0:
                all_resolved = False
                break
        
        # 检查是否达到最大时间步
        time_limit_reached = self.current_time_step >= config.SIMULATION_TIME - 1
        
        return all_resolved or time_limit_reached
    
    def step(self, rescuer_idx, action):
        """执行指定救援人员的动作并返回结果"""
        # 记录旧状态和当前灾情信息以计算奖励
        old_state = self.env.rescuers[rescuer_idx].copy() if rescuer_idx < len(self.env.rescuers) else {}
        old_disasters = {pos: disaster.copy() for pos, disaster in self.env.disasters.items()}
        
        # 将动作转换为目标位置
        target_pos = None
        if action > 0:  # 非0动作表示前往某个位置
            action_idx = action - 1
            x = action_idx // self.env.GRID_SIZE
            y = action_idx % self.env.GRID_SIZE
            target_pos = (x, y)
            
            # 如果目标位置有灾情，则分配任务
            if target_pos in self.env.disasters:
                self.env.rescuers[rescuer_idx]["target"] = target_pos
            else:
                # 如果目标位置没有灾情，则清除当前任务
                if "target" in self.env.rescuers[rescuer_idx]:
                    del self.env.rescuers[rescuer_idx]["target"]
        
        # 执行救援
        from src.core.rescue_execution import execute_rescue
        execute_rescue(self.env.rescuers, self.env.disasters, self.env.GRID_SIZE, current_time_step=self.current_time_step)
        
        # 计算奖励
        from src.rl.rl_util import calculate_reward
        reward, reward_info = calculate_reward(self.env, rescuer_idx, old_state, old_disasters)
        
        # 获取新状态
        next_state = self.get_state_for_rescuer(rescuer_idx)
        
        # 检查是否结束
        done = self.is_episode_done()
        
        # 构建信息字典
        info = {
            "success": False,
            "response_time": 0
        }
        
        # 如果有目标且目标是灾情点且灾情已解决，则表示成功
        if "target" in self.env.rescuers[rescuer_idx]:
            target = self.env.rescuers[rescuer_idx]["target"]
            if target in old_disasters and target in self.env.disasters:
                old_disaster = old_disasters[target]
                current_disaster = self.env.disasters[target]
                
                # 如果灾情解决或减轻
                if old_disaster["rescue_needed"] > current_disaster["rescue_needed"]:
                    info["success"] = True
                    # 计算响应时间 - 使用time_step而不是start_time
                    if "time_step" in current_disaster:
                        response_time = self.current_time_step - current_disaster["time_step"]
                        # 确保响应时间为正数
                        if response_time <= 0:
                            print(f"警告：检测到负响应时间，current_time_step={self.current_time_step}, disaster_time_step={current_disaster['time_step']}")
                            response_time = 1  # 确保至少为1
                        info["response_time"] = response_time
                    # 如果找不到time_step，则使用一个合理的默认值
                    else:
                        # 确保有一个有效的响应时间
                        info["response_time"] = max(1, min(10, self.current_time_step))  # 使用1到10之间的合理值
                        print(f"未找到time_step，使用默认响应时间={info['response_time']}")
        
        # 更新当前时间步
        self.current_time_step += 1
        
        return next_state, reward, done, info


# 训练MARL系统
def train_marl(env, controller, num_episodes, max_steps, with_verbose=False, save_freq=10):
    """
    自定义训练循环，用于MARL训练
    """
    """
    自定义训练循环，用于MARL训练
    """
    print("开始MARL训练过程...")
    print("-------------------------------------------")
    print("训练分为三个阶段：")
    print(" - 阶段1：灾难初期，生成概率约0.5，维持20-50个灾难点（少于20个自动补充到20个，多于50个自动减少到50个）")
    print(" - 阶段2：灾难中期，生成概率约0.3，维持5-20个灾难点（少于5个自动补充到5个，多于20个自动减少到20个）")
    print(" - 阶段3：灾难后期，生成概率约0.1，维持1-5个灾难点（少于1个自动补充到1个，多于5个自动减少到5个）")
    print("-------------------------------------------")
    
    # 存储每个回合的奖励，用于计算平均奖励
    all_rewards = []
    success_rates = []
    response_times = []
    losses = []  # 添加损失追踪
    
    # 新增：用于高级可视化的环境快照列表
    env_snapshots = []
    
    # 开始训练
    for episode in range(num_episodes):
        total_reward = 0
        success_count = 0
        total_response_time = 0
        disaster_count = 0
        episode_loss = 0  # 记录本回合的平均损失
        
        # 重置环境 - 创建新的环境实例而不是调用reset方法
        if episode > 0:  # 只有在第二轮开始时才需要重置，因为第一轮已经有初始环境
            env = Environment(verbose=False)  # 使用无输出版本
            # 更新环境缓存
            if hasattr(controller, "_env_cache"):
                controller._env_cache = [env]
        
        # 调整探索率
        epsilon_progress = min(1.0, episode / (0.8 * num_episodes))
        epsilon_start = controller.epsilon_start  # 使用控制器的epsilon_start
        epsilon_end = controller.epsilon_end  # 使用控制器的epsilon_end
        epsilon = max(epsilon_end, epsilon_start - epsilon_progress * (epsilon_start - epsilon_end))
        controller.epsilon = epsilon
        
        # 每步保存环境快照，但不立即可视化
        for step in range(max_steps):
            if with_verbose:
                print(f"  步骤 {step+1}/{max_steps}...")
            
            # 获取每步开始时的灾难点数量（用于日志）
            pre_adjust_disasters = len(env.disasters) if hasattr(env, "disasters") else len(env.disaster_locations) if hasattr(env, "disaster_locations") else 0
                
            # 调整灾难设置（不再使用详细输出）
            adjust_disaster_settings(env, step, max_steps, verbose=False)
            
            # 获取调整后的灾难点数量（用于日志）
            post_adjust_disasters = len(env.disasters) if hasattr(env, "disasters") else len(env.disaster_locations) if hasattr(env, "disaster_locations") else 0
            
            # 每10步输出当前灾难点数量
            if step % 10 == 0:
                # 根据环境接口获取灾难点数量
                current_disaster_points = post_adjust_disasters
                
                # 根据环境接口统计不同等级的灾难点
                if hasattr(env, "disasters"):
                    high_level = sum(1 for d in env.disasters.values() if d["level"] >= 9)
                    medium_level = sum(1 for d in env.disasters.values() if 7 <= d["level"] < 9)
                    low_level = sum(1 for d in env.disasters.values() if d["level"] < 7)
                else:
                    # 如果环境接口不兼容，则设置为0
                    high_level = medium_level = low_level = 0
                
                # 简化输出，不再显示变化情况
                print(f"{Colors.CYAN}[步骤 {step}/{max_steps}] 当前灾难点: {current_disaster_points} " +
                      f"(高风险: {Colors.RED}{high_level}{Colors.CYAN}, " +
                      f"中风险: {Colors.YELLOW}{medium_level}{Colors.CYAN}, " +
                      f"低风险: {Colors.GREEN}{low_level}{Colors.ENDC})")
            
            # 更新灾难状态（使用无调试输出模式）
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                if hasattr(env, 'update_disasters_silent'):
                    env.update_disasters_silent(current_time_step=env.current_time_step)
                else:
                    env.update_disasters(current_time_step=env.current_time_step)
            
            # 增加环境的当前时间步，确保时间正确推进
            env.current_time_step = env.current_time_step + 1 if hasattr(env, 'current_time_step') else step
            
            # 根据环境接口获取当前灾难点数量
            current_disasters = len(env.disasters) if hasattr(env, "disasters") else len(env.disaster_locations) if hasattr(env, "disaster_locations") else 0
            disaster_count = max(disaster_count, current_disasters)
            
            # 遍历每个救援者智能体
            for rescuer_idx in range(config.NUM_RESCUERS):
                if with_verbose:
                    print(f"    处理救援者 {rescuer_idx+1}/{config.NUM_RESCUERS}...")
                    
                # 获取当前状态
                state = env.get_state_for_rescuer(rescuer_idx)
                
                # 选择动作
                action = controller.select_action(state, rescuer_idx)
                
                # 执行动作并获取奖励（使用无调试输出模式）
                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                    next_state, reward, done, info = env.step(rescuer_idx, action)
                
                # 处理奖励信息
                total_reward += reward
                if 'success' in info and info['success']:
                    success_count += 1
                if 'response_time' in info and info['response_time'] > 0:  # 确保响应时间有效
                    total_response_time += info['response_time']
                    
                # 存储转换到经验回放缓冲区
                controller.store_transition(state, action, reward, next_state, done, rescuer_idx)
            
            # 更新神经网络
            loss = controller.update_agents()
            episode_loss += loss  # 累计损失
            
            # 保存环境快照 - 每一步都保存
            env_copy = copy.deepcopy(env)
            
            # 计算当前步骤的成功率（临时计算用于快照记录）
            current_success_rate = success_count / max(1, disaster_count)
            
            # 创建快照字典
            snapshot = {
                "env": env_copy,
                "time_step": episode * max_steps + step,  # 计算总时间步
                "success_rate": current_success_rate,
                "episode": episode + 1,
                "step": step + 1
            }
            
            # 添加到快照列表
            env_snapshots.append(snapshot)
            
            # 如果训练过程结束，跳出循环
            if env.is_episode_done():
                if with_verbose:
                    print("回合结束条件满足，提前结束本回合")
                break
        
        # 计算平均损失 - 确保不为零
        steps_completed = min(step + 1, max_steps)  # 使用实际完成的步数
        avg_loss = episode_loss / steps_completed if steps_completed > 0 else 0
        
        # 不再对损失进行裁剪
        losses.append(avg_loss)
        
        # 计算平均奖励和成功率
        avg_reward = total_reward / config.NUM_RESCUERS if config.NUM_RESCUERS > 0 else 0
        success_rate = success_count / max(1, disaster_count)  # 确保分母非零
        
        # 计算平均响应时间 - 修复计算逻辑
        avg_response_time = total_response_time / max(1, success_count)  # 确保分母非零
        
        all_rewards.append(avg_reward)
        success_rates.append(success_rate)
        response_times.append(avg_response_time)
        
        # 每隔一定回合保存模型
        if (episode + 1) % save_freq == 0:
            controller.save_models(f"model_episode_{episode+1}")
            print(f"[进度 {episode+1}/{num_episodes}] 已保存模型 (完成: {(episode+1)/num_episodes*100:.1f}%)")
        
        # 每轮输出一次综合信息 - 结合了之前分散的输出
        print(f"[轮次 {episode+1}/{num_episodes}] 探索率: {epsilon:.4f}, 平均奖励: {avg_reward:.2f}, " +
              f"成功率: {success_rate:.2f}, 平均响应时间: {avg_response_time:.2f}秒, 平均损失: {avg_loss:.4f}")
        
        # 每轮都输出本轮训练详细数据
        print(f"\n----- 本轮训练详细数据 -----")
        print(f"• 灾难点数量: {disaster_count}")
        print(f"• 成功救援次数: {success_count}")
        print(f"• 总奖励: {total_reward:.2f}")
        print(f"• 训练步数: {steps_completed}")
        
        # 每10个回合输出一次最近100回合的统计信息
        if (episode + 1) % 10 == 0 or episode == num_episodes - 1:
            recent_rewards = all_rewards[-100:] if len(all_rewards) >= 100 else all_rewards
            recent_success = success_rates[-100:] if len(success_rates) >= 100 else success_rates
            recent_response = response_times[-100:] if len(response_times) >= 100 else response_times
            recent_losses = losses[-100:] if len(losses) >= 100 else losses
            
            print("\n========== 最近训练统计信息 ==========")
            print(f"最近{len(recent_rewards)}回合统计:")
            print(f"• 平均奖励: {np.mean(recent_rewards):.2f}")
            print(f"• 平均成功率: {np.mean(recent_success):.2f}")
            print(f"• 平均响应时间: {np.mean(recent_response):.2f}秒")
            print(f"• 平均损失: {np.mean(recent_losses):.4f}")
            print(f"• 当前探索率: {epsilon:.4f}")
            print(f"• 最高成功率: {max(recent_success) if recent_success else 0:.2f}")
            
            # 添加奖励分解信息（如果可用）
            if hasattr(controller, "get_reward_stats"):
                reward_stats = controller.get_reward_stats(reset=False)
                print(f"\n奖励分解:")
                for reward_type, stats in reward_stats.items():
                    if stats["total"] != 0:  # 只显示非零奖励
                        print(f"• {reward_type}: 总计={stats['total']:.2f}, 平均={stats['avg']:.2f}")
            
            print("==================================\n")
        
        # 每轮结束时保存元数据，简化逻辑，移除可视化部分
        print(f"\n[元数据] 正在保存第 {episode+1} 轮的训练元数据...")

        # 创建元数据保存目录结构
        metadata_dir = os.path.join(SAVE_DIR, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        # 创建该轮的元数据文件名（确保不会覆盖）
        episode_meta_file = os.path.join(metadata_dir, f"episode_{episode+1:04d}.json")

        # 提取当前轮的快照元数据
        episode_meta = []
        for snapshot in env_snapshots:
            if snapshot["episode"] == episode + 1:
                # 保存元数据
                meta = {
                    "time_step": snapshot["time_step"],
                    "success_rate": snapshot["success_rate"],
                    "episode": snapshot["episode"],
                    "step": snapshot["step"],
                    # 添加更多状态信息
                    "disaster_count": disaster_count,
                    "success_count": success_count,
                    "total_reward": total_reward,
                    "avg_reward": avg_reward,
                    "avg_response_time": avg_response_time
                }
                episode_meta.append(meta)

        # 保存元数据（按步骤排序）
        episode_meta.sort(key=lambda x: x["step"])
        with open(episode_meta_file, 'w') as f:
            json.dump({
                "metadata": episode_meta,
                "env_config": {
                    "grid_size": env.GRID_SIZE,
                    "num_rescuers": len(env.rescuers),
                    "max_steps": max_steps
                },
                "metrics": {
                    "rewards": all_rewards[:episode+1],
                    "success_rates": success_rates[:episode+1],
                    "response_times": response_times[:episode+1],
                    "losses": losses[:episode+1] if losses else []
                }
            }, f)

        print(f"[元数据] 第 {episode+1} 轮训练元数据已保存到: {episode_meta_file}")

        # 保存训练环境快照，用于后续可能的可视化
        snapshot_dir = os.path.join(SAVE_DIR, "snapshots", f"episode_{episode+1:04d}")
        os.makedirs(snapshot_dir, exist_ok=True)

        # 保存最后一步的环境快照
        if len(env_snapshots) > 0:
            # 找到当前轮的最后一个快照
            last_snapshot = None
            for snapshot in reversed(env_snapshots):
                if snapshot["episode"] == episode + 1:
                    last_snapshot = snapshot
                    break
            
            if last_snapshot:
                # 保存仅包含必要信息的环境快照
                snapshot_file = os.path.join(snapshot_dir, "final_state.pkl")
                with open(snapshot_file, 'wb') as f:
                    pickle.dump(last_snapshot, f)
                print(f"[快照] 已保存环境状态快照到: {snapshot_file}")

        # 只保留最新一轮的环境快照，释放内存
        if episode > 0:  # 第一轮之后才清理
            # 筛选出当前轮的快照，释放之前轮的快照
            env_snapshots = [s for s in env_snapshots if s["episode"] == episode + 1]
            print(f"[内存] 已清理旧轮次环境快照，当前保留 {len(env_snapshots)} 个快照")
    
    return all_rewards, success_rates, response_times




# 评估MARL系统
def evaluate_marl(env, episodes=5, max_steps=config.SIMULATION_TIME):
    """评估MARL系统性能"""
    # 创建MARL控制器
    marl = MARLController(
        grid_size=env.GRID_SIZE,
        num_rescuers=len(env.rescuers),
        hidden_dim=config.MARL_CONFIG["hidden_dim"],
        lr=config.MARL_CONFIG["learning_rate"],
        gamma=config.MARL_CONFIG["gamma"]
    )
    
    # 加载模型
    if not marl.load_models():
        print("无法加载模型，使用随机策略进行评估")
    
    # 创建强化学习环境包装
    rl_env = RescueEnvironment(env)
    
    # 记录评估数据
    all_success_rates = []
    all_avg_response_times = []
    
    for episode in range(episodes):
        # 重置环境
        env = rl_env.reset()
        
        for step in range(max_steps):
            # 选择动作（评估模式，不使用探索）
            actions = marl.select_actions(env.rescuers, env.disasters, training=False)
            
            # 执行动作
            env, _, _, done, _ = rl_env.step(actions, step)
            
            if done:
                break
        
        # 计算本轮评估的统计数据
        from src.utils.stats import calculate_rescue_success_rate, calculate_average_response_time
        
        success_rate = calculate_rescue_success_rate(env.disasters, window=config.STATS_WINDOW_SIZE)
        avg_response_time = calculate_average_response_time(env.disasters)
        
        all_success_rates.append(success_rate)
        all_avg_response_times.append(avg_response_time)
        
        print(f"评估轮次 {episode+1}/{episodes} | "
              f"救援成功率: {success_rate:.2f} | "
              f"平均响应时间: {avg_response_time:.2f}")
    
    # 计算平均性能
    avg_success_rate = sum(all_success_rates) / len(all_success_rates)
    avg_response_time = sum(all_avg_response_times) / len(all_avg_response_times)
    
    print(f"\n评估结果:")
    print(f"平均救援成功率: {avg_success_rate:.2f}")
    print(f"平均响应时间: {avg_response_time:.2f}")
    
    return avg_success_rate, avg_response_time 