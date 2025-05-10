import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from src.core import config

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
    def __init__(self, grid_size, num_rescuers, hidden_dim=128, lr=0.001, gamma=0.99):
        self.grid_size = grid_size
        self.num_rescuers = num_rescuers
        self.gamma = gamma
        
        # 设置设备
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device_output = False  # 添加标志，避免多次输出设备信息
        #print(f"使用设备: {self._device}")
        
        # 状态空间：
        # 1. 救援人员位置(x,y)
        # 2. 救援人员是否有任务分配
        # 3. 救援人员的移动速度
        # 4. 每个可能的目标灾情点的信息(x,y,level,rescue_needed)
        # 5. 每个灾情点的当前救援资源分配情况
        # 状态维度计算: 救援人员自身状态(4) + 灾情信息(grid_size*grid_size*5)
        self.state_dim = 4 + grid_size * grid_size * 5
        
        # 动作空间：
        # 0: 不动
        # 1~grid_size*grid_size: 前往对应格子坐标的灾情点
        self.action_dim = 1 + grid_size * grid_size
        
        # 为每个救援人员创建一个智能体
        self.agents = [RescuerAgent(self.state_dim, self.action_dim, hidden_dim, device=self._device) for _ in range(num_rescuers)]
        self.optimizers = [optim.Adam(agent.parameters(), lr=lr) for agent in self.agents]
        
        # 经验回放缓冲区
        self.replay_buffers = [ExperienceReplay(config.MARL_CONFIG["replay_buffer_size"]) for _ in range(num_rescuers)]
        
        # 目标网络
        self.target_agents = [RescuerAgent(self.state_dim, self.action_dim, hidden_dim, device=self._device) for _ in range(num_rescuers)]
        for i in range(num_rescuers):
            self.target_agents[i].load_state_dict(self.agents[i].state_dict())
        
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
    
    def select_action(self, state, rescuer_idx):
        """为单个救援人员选择动作"""
        # 检查rescuer_idx是否超出范围
        if rescuer_idx >= self.num_rescuers:
            print(f"警告: rescuer_idx ({rescuer_idx}) 超出了范围 ({self.num_rescuers})")
            return 0  # 返回不动作
            
        # 确保state是张量且在正确设备上
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
            
        # 使用epsilon-greedy策略选择动作
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                _, policy_logits = self.agents[rescuer_idx](state)
                return torch.argmax(policy_logits).item()
    
    def select_actions(self, rescuers, disasters, training=False):
        """为所有救援人员选择动作，兼容旧版本API"""
        actions = []
        
        # 更新探索率
        if training:
            self.steps_done += 1
        
        for i in range(self.num_rescuers):
            state = self.build_state(i, rescuers, disasters)
            # 使用select_action方法选择动作
            action = self.select_action(state, i)
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
    
    def calculate_reward(self, rescuer_idx, old_state, old_disasters):
        """
        计算单个救援人员的奖励
        
        奖励设计:
        1. 成功完成救援任务: +COMPLETION_REWARD
        2. 前往高优先级灾情点: +level*PRIORITY_FACTOR (优先处理高风险灾情)
        3. 与其他救援人员协调避免重复: +COORDINATION_REWARD (如果该灾情点没有其他人前往)
        4. 救援进度奖励: +progress*PROGRESS_FACTOR (根据救援进度增加的奖励)
        5. 时间惩罚: -TIME_PENALTY (每个时间步的轻微惩罚，鼓励快速行动)
        """
        # 奖励系数 - 可以在这里调整奖励大小
        COMPLETION_REWARD = 10.0    # 完成救援奖励
        PRIORITY_FACTOR = 0.1       # 优先级因子 (乘以灾情等级)
        COORDINATION_REWARD = 2.0   # 协调奖励
        PROGRESS_FACTOR = 1.0       # 进度奖励因子 (乘以救援进度)
        TIME_PENALTY = 0.1          # 时间惩罚
        
        # 确保rescuer_idx在有效范围内
        if rescuer_idx >= len(self.env.rescuers):
            return 0, {"completion_reward": 0, "priority_reward": 0, 
                     "coordination_reward": 0, "progress_reward": 0, "time_penalty": 0}
            
        rescuer = self.env.rescuers[rescuer_idx]
        reward = 0
        
        # 初始化奖励明细
        reward_info = {
            "completion_reward": 0.0,
            "priority_reward": 0.0,
            "coordination_reward": 0.0,
            "progress_reward": 0.0,
            "time_penalty": -TIME_PENALTY  # 基础时间惩罚
        }
        
        # 累加时间惩罚
        reward += reward_info["time_penalty"]
        
        # 如果救援人员有目标
        if "target" in rescuer and rescuer["target"] is not None:
            target = rescuer["target"]
            
            # 如果目标是有效的灾情点
            if target in self.env.disasters:
                current_disaster = self.env.disasters[target]
                old_disaster = old_disasters.get(target, None)
                
                # 奖励1: 根据灾情等级给予奖励
                priority_reward = current_disaster["level"] * PRIORITY_FACTOR
                reward_info["priority_reward"] = priority_reward
                reward += priority_reward
                
                # 奖励2: 协调奖励 - 检查是否有其他救援人员前往同一目标
                other_targeting = False
                for i, other_rescuer in enumerate(self.env.rescuers):
                    if i != rescuer_idx and "target" in other_rescuer and other_rescuer["target"] == target:
                        other_targeting = True
                        break
                
                if not other_targeting:
                    coordination_reward = COORDINATION_REWARD
                    reward_info["coordination_reward"] = coordination_reward
                    reward += coordination_reward
                
                # 奖励3: 根据救援进度奖励
                if old_disaster:
                    # 如果旧灾情存在，计算救援进度奖励
                    if old_disaster["rescue_needed"] > current_disaster["rescue_needed"]:
                        progress = old_disaster["rescue_needed"] - current_disaster["rescue_needed"]
                        progress_reward = progress * PROGRESS_FACTOR
                        reward_info["progress_reward"] = progress_reward
                        reward += progress_reward
                
                # 奖励4: 如果灾情已解决，给予大奖励
                if current_disaster["rescue_needed"] <= 0:
                    completion_reward = COMPLETION_REWARD
                    reward_info["completion_reward"] = completion_reward
                    reward += completion_reward
        
        return reward, reward_info
    
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


class RescueEnvironment:
    """强化学习环境封装，用于与现有的救援环境交互"""
    def __init__(self, env):
        self.env = env
        self.grid_size = env.GRID_SIZE
        self.num_rescuers = len(env.rescuers)
        self.current_time_step = 0
        
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
                grid_size=self.grid_size,
                num_rescuers=self.num_rescuers
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
            x = action_idx // self.grid_size
            y = action_idx % self.grid_size
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
        reward, reward_info = self.calculate_reward(rescuer_idx, old_state, old_disasters)
        
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
    
    def calculate_reward(self, rescuer_idx, old_state, old_disasters):
        """
        计算单个救援人员的奖励
        
        奖励设计:
        1. 成功完成救援任务: +COMPLETION_REWARD
        2. 前往高优先级灾情点: +level*PRIORITY_FACTOR (优先处理高风险灾情)
        3. 与其他救援人员协调避免重复: +COORDINATION_REWARD (如果该灾情点没有其他人前往)
        4. 救援进度奖励: +progress*PROGRESS_FACTOR (根据救援进度增加的奖励)
        5. 时间惩罚: -TIME_PENALTY (每个时间步的轻微惩罚，鼓励快速行动)
        """
        # 奖励系数 - 可以在这里调整奖励大小
        COMPLETION_REWARD = 10.0    # 完成救援奖励
        PRIORITY_FACTOR = 0.1       # 优先级因子 (乘以灾情等级)
        COORDINATION_REWARD = 2.0   # 协调奖励
        PROGRESS_FACTOR = 1.0       # 进度奖励因子 (乘以救援进度)
        TIME_PENALTY = 0.1         # 时间惩罚
        
        # 确保rescuer_idx在有效范围内
        if rescuer_idx >= len(self.env.rescuers):
            return 0, {"completion_reward": 0, "priority_reward": 0, 
                     "coordination_reward": 0, "progress_reward": 0, "time_penalty": 0}
            
        rescuer = self.env.rescuers[rescuer_idx]
        reward = 0
        
        # 初始化奖励明细
        reward_info = {
            "completion_reward": 0.0,
            "priority_reward": 0.0,
            "coordination_reward": 0.0,
            "progress_reward": 0.0,
            "time_penalty": -TIME_PENALTY  # 基础时间惩罚
        }
        
        # 累加时间惩罚
        reward += reward_info["time_penalty"]
        
        # 如果救援人员有目标
        if "target" in rescuer and rescuer["target"] is not None:
            target = rescuer["target"]
            
            # 如果目标是有效的灾情点
            if target in self.env.disasters:
                current_disaster = self.env.disasters[target]
                old_disaster = old_disasters.get(target, None)
                
                # 奖励1: 根据灾情等级给予奖励
                priority_reward = current_disaster["level"] * PRIORITY_FACTOR
                reward_info["priority_reward"] = priority_reward
                reward += priority_reward
                
                # 奖励2: 协调奖励 - 检查是否有其他救援人员前往同一目标
                other_targeting = False
                for i, other_rescuer in enumerate(self.env.rescuers):
                    if i != rescuer_idx and "target" in other_rescuer and other_rescuer["target"] == target:
                        other_targeting = True
                        break
                
                if not other_targeting:
                    coordination_reward = COORDINATION_REWARD
                    reward_info["coordination_reward"] = coordination_reward
                    reward += coordination_reward
                
                # 奖励3: 根据救援进度奖励
                if old_disaster:
                    # 如果旧灾情存在，计算救援进度奖励
                    if old_disaster["rescue_needed"] > current_disaster["rescue_needed"]:
                        progress = old_disaster["rescue_needed"] - current_disaster["rescue_needed"]
                        progress_reward = progress * PROGRESS_FACTOR
                        reward_info["progress_reward"] = progress_reward
                        reward += progress_reward
                
                # 奖励4: 如果灾情已解决，给予大奖励
                if current_disaster["rescue_needed"] <= 0:
                    completion_reward = COMPLETION_REWARD
                    reward_info["completion_reward"] = completion_reward
                    reward += completion_reward
        
        return reward, reward_info


# 训练MARL系统
def train_marl(env, num_episodes=100, max_steps=config.SIMULATION_TIME):
    """训练MARL系统"""
    # 创建MARL控制器
    marl = MARLController(
        grid_size=env.GRID_SIZE,
        num_rescuers=len(env.rescuers),
        hidden_dim=config.MARL_CONFIG["hidden_dim"],
        lr=config.MARL_CONFIG["learning_rate"],
        gamma=config.MARL_CONFIG["gamma"]
    )
    
    # 创建强化学习环境包装
    rl_env = RescueEnvironment(env)
    
    # 记录训练数据
    all_episode_rewards = []
    all_success_rates = []
    all_avg_response_times = []
    
    print("开始训练MARL系统...")
    for episode in range(num_episodes):
        # 重置环境
        env = rl_env.reset()
        episode_rewards = [0] * len(env.rescuers)
        
        for step in range(max_steps):
            # 选择动作
            actions = marl.select_actions(env.rescuers, env.disasters, training=True)
            
            # 记录当前状态
            states = [marl.build_state(i, env.rescuers, env.disasters) for i in range(len(env.rescuers))]
            
            # 执行动作
            env, rewards, reward_infos, done, _ = rl_env.step(actions, step)
            
            # 记录新状态
            next_states = [marl.build_state(i, env.rescuers, env.disasters) for i in range(len(env.rescuers))]
            
            # 存储经验和跟踪奖励
            for i in range(len(env.rescuers)):
                marl.store_experience(
                    i, states[i], actions[i], rewards[i], next_states[i], done
                )
                marl.track_rewards(reward_infos[i])
            
            # 更新智能体
            loss = marl.update_agents()
            
            # 累计奖励
            for i in range(len(env.rescuers)):
                episode_rewards[i] += rewards[i]
            
            if done:
                break
        
        # 计算本轮训练的统计数据
        from src.utils.stats import calculate_rescue_success_rate, calculate_average_response_time
        
        success_rate = calculate_rescue_success_rate(env.disasters, window=config.STATS_WINDOW_SIZE)
        avg_response_time = calculate_average_response_time(env.disasters)
        
        avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
        all_episode_rewards.append(avg_episode_reward)
        all_success_rates.append(success_rate)
        all_avg_response_times.append(avg_response_time)
        
        print(f"Episode {episode+1}/{num_episodes} | "
              f"Avg Reward: {avg_episode_reward:.2f} | "
              f"Success Rate: {success_rate:.2f} | "
              f"Avg Response Time: {avg_response_time:.2f} | "
              f"Epsilon: {marl.epsilon:.2f}")
        
        # 每10轮保存一次模型
        if (episode + 1) % 10 == 0:
            marl.save_models()
            
        # 每100轮输出奖励统计
        if (episode + 1) % 100 == 0:
            reward_stats = marl.get_reward_stats()
            print("\n=== 最近100轮奖励统计 ===")
            print(f"完成任务奖励: 总计={reward_stats['completion_reward']['total']:.2f}, 平均={reward_stats['completion_reward']['avg']:.2f}")
            print(f"高优先级任务奖励: 总计={reward_stats['priority_reward']['total']:.2f}, 平均={reward_stats['priority_reward']['avg']:.2f}")
            print(f"协调奖励: 总计={reward_stats['coordination_reward']['total']:.2f}, 平均={reward_stats['coordination_reward']['avg']:.2f}")
            print(f"进度奖励: 总计={reward_stats['progress_reward']['total']:.2f}, 平均={reward_stats['progress_reward']['avg']:.2f}")
            print(f"时间惩罚: 总计={reward_stats['time_penalty']['total']:.2f}, 平均={reward_stats['time_penalty']['avg']:.2f}")
            print("=========================\n")
    
    # 训练结束，保存最终模型
    marl.save_models()
    
    return marl, all_episode_rewards, all_success_rates, all_avg_response_times


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