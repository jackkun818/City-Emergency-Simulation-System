import numpy as np#1
import random
import time
import random
from . import config  # 导入配置文件

# 这些常量将被 config 中的参数替代
GRID_SIZE = 10
NUM_DISASTERS = 5
NUM_RESCUERS = 3


class Environment:
    def __init__(self, grid_size=None, num_rescuers=None, verbose=True, rescuers_data=None, training_mode=False):
        # 使用 config 中的参数，如果有传入参数则使用传入的参数
        self.GRID_SIZE = grid_size if grid_size is not None else config.get_config_param("grid_size")
        self.num_rescuers = num_rescuers if num_rescuers is not None else config.get_config_param("num_rescuers")
        self.rescuers = []
        self.disasters = {}
        self.current_time_step = 0  # 添加当前时间步属性
        self.training_mode = training_mode  # 训练模式标识

        # 打印灾难规模信息
        if verbose:
            if config.DISASTER_SCALE == 3:
                print(f"使用自定义灾难规模: 网格大小={self.GRID_SIZE}, 灾情生成概率={config.get_config_param('disaster_spawn_rate')}")
            else:
                preset = config.DISASTER_PRESETS[config.DISASTER_SCALE]
                print(f"使用预设灾难规模: {preset['name']}, 网格大小={preset['grid_size']}, 灾情生成概率={preset['disaster_spawn_rate']}")
            
            # 打印救援人员信息（独立于灾难规模）
            print(f"救援人员数量: {self.num_rescuers}")
        
        # 如果提供了救援人员数据，直接使用它
        if rescuers_data is not None:
            self.rescuers = rescuers_data
            self.num_rescuers = len(rescuers_data)
            if verbose:
                print(f"使用预定义的救援人员数据，数量: {self.num_rescuers}")
        else:
            self.initialize_rescuers()

    def initialize_rescuers(self):
        """ 初始化救援人员并设置差异化的能力和速度 """
        for i in range(self.num_rescuers):
            # 随机生成每个救援人员的能力和速度
            capacity = random.randint(1, 3)  # 救援能力在1到3之间
            speed = random.randint(1, 3)  # 移动速度在1到3之间默认系统时间生成的

            self.rescuers.append({
                "id": i,
                "position": (0, 0),  # 初始位置可以是任意的
                "active_time": 0,
                "capacity": capacity,
                "speed": speed
            })

    def update_disasters(self, current_time_step=None):
        """ 
        模拟灾情的出现和变化
        :param current_time_step: 当前的时间步，用于记录灾情点的创建时间
        """
        # 更新红叉计数器
        for pos, disaster in list(self.disasters.items()):
            # 如果红叉计数器大于0，则递减
            if disaster.get("show_red_x", 0) > 0:
                disaster["show_red_x"] -= 1
                if disaster["show_red_x"] == 0:
                    print(f"📍 灾情点 {pos} 红叉显示时间结束，不再显示")

            # 记录前一个时间步的灾情等级，用于计算进度奖励
            if not disaster.get("frozen_level", False) and not disaster.get("frozen_rescue", False):
                disaster["previous_level"] = disaster["level"]

        # 获取当前时间步的实际灾情生成概率
        if current_time_step is not None:
            # 优先使用adjust_disaster_settings设置的概率
            if hasattr(self, 'disaster_gen_prob'):
                actual_spawn_rate = self.disaster_gen_prob
            else:
                actual_spawn_rate = config.get_actual_spawn_rate(current_time_step)
            if current_time_step % 10 == 0:  # 每10个时间步打印一次概率
                print(f"当前时间步: {current_time_step}, 灾情生成概率: {actual_spawn_rate:.3f}")
        else:
            # 如果没有提供时间步，使用基础概率
            if hasattr(self, 'disaster_gen_prob'):
                actual_spawn_rate = self.disaster_gen_prob
            else:
                actual_spawn_rate = config.get_config_param("disaster_spawn_rate")

        # 只在训练模式下启用数量上限控制
        if self.training_mode:
            # 统计当前活跃的灾难点数量（只计算需要救援的点）
            active_disaster_count = sum(1 for disaster in self.disasters.values() if disaster.get("rescue_needed", 0) > 0)
            max_disasters = self._get_disaster_limit(current_time_step)
            
            # 获取下限值（与adjust_disaster_settings保持一致）
            max_steps = getattr(self, '_max_steps', 300)
            phase1_end = int(max_steps * 2 / 3)
            phase2_end = int(max_steps * 5 / 6)
            
            if current_time_step is None or current_time_step < phase1_end:
                min_disasters = 20  # 初期阶段下限
            elif current_time_step < phase2_end:
                min_disasters = 5   # 中期阶段下限
            else:
                min_disasters = 1   # 后期阶段下限
            
            # 如果活跃灾难点不足，强制添加以达到下限
            if active_disaster_count < min_disasters:
                shortage = min_disasters - active_disaster_count
                added_count = 0
                max_add_attempts = shortage * 20  # 增加尝试次数
                
                for _ in range(max_add_attempts):
                    if added_count >= shortage:
                        break
                        
                    x, y = np.random.randint(0, self.GRID_SIZE, size=2)
                    if (x, y) not in self.disasters:
                        # 生成新的灾难点
                        level = np.random.randint(5, 11)  # 5-10之间
                        
                        # 统一rescue_needed的生成范围，与部署模式保持一致
                        if level <= 6:
                            rescue_needed = np.random.randint(5, 7)  # 5-6之间  
                        elif level <= 8:
                            rescue_needed = np.random.randint(7, 9)  # 7-8之间
                        else:
                            rescue_needed = np.random.randint(9, 11)  # 9-10之间
                        
                        # 新灾情点加入初始时间和时间步信息
                        self.disasters[(x, y)] = {
                            "level": level,
                            "rescue_needed": rescue_needed,
                            "time_step": self.current_time_step if current_time_step is None else current_time_step,
                            "frozen_level": False,
                            "frozen_rescue": False,
                            "rescue_success": False,
                            "show_red_x": 0
                        }
                        added_count += 1
            
            # 只有在未达到活跃灾难点上限时才生成新的灾情点
            elif active_disaster_count < max_disasters:
                # 增加灾难点生成数量，确保有足够的挑战性
                target_spawn_count = int(actual_spawn_rate * self.GRID_SIZE / 5)  # 从20改为5，增加生成数量
                if target_spawn_count < 2 and actual_spawn_rate > 0.1:
                    target_spawn_count = 2  # 确保至少尝试生成2个
                
                # 限制生成数量不超过上限
                max_new_disasters = max_disasters - active_disaster_count
                target_spawn_count = min(target_spawn_count, max_new_disasters)
                
                # 随机选择位置生成灾难点
                new_disasters_count = 0
                attempts = 0
                max_attempts = target_spawn_count * 10  # 最多尝试10倍的次数
                
                while new_disasters_count < target_spawn_count and attempts < max_attempts:
                    x, y = np.random.randint(0, self.GRID_SIZE, size=2)
                    attempts += 1
                    
                    # 如果位置已被占用，跳过
                    if (x, y) in self.disasters:
                        continue
                    
                    # 生成新的灾难点
                    level = np.random.randint(5, 11)  # 5-10之间
                    
                    # 统一rescue_needed的生成范围，与部署模式保持一致
                    if level <= 6:
                        rescue_needed = np.random.randint(5, 7)  # 5-6之间  
                    elif level <= 8:
                        rescue_needed = np.random.randint(7, 9)  # 7-8之间
                    else:
                        rescue_needed = np.random.randint(9, 11)  # 9-10之间
                    
                    # 新灾情点加入初始时间和时间步信息
                    self.disasters[(x, y)] = {
                        "level": level,
                        "rescue_needed": rescue_needed,
                        "time_step": self.current_time_step if current_time_step is None else current_time_step,  # 记录灾情点创建的时间步
                        "frozen_level": False,  # 初始状态为未冻结
                        "frozen_rescue": False,  # 初始状态为未冻结
                        "rescue_success": False,  # 初始状态为未救援成功
                        "show_red_x": 0  # 红叉显示计数器，0表示不显示
                    }
                    new_disasters_count += 1
        else:
            # 非训练模式，不启用数量上限控制
            # 随机生成新的灾情点
            for _ in range(int(actual_spawn_rate * self.GRID_SIZE)):
                x, y = np.random.randint(0, self.GRID_SIZE, size=2)
                if (x, y) not in self.disasters:
                    # 先生成level，范围5-10
                    level = np.random.randint(5, 11)  # 注意上限改为11，使范围包含10
                    
                    # 统一rescue_needed的生成范围，与部署模式保持一致
                    if level <= 6:
                        rescue_needed = np.random.randint(5, 7)  # 5-6之间  
                    elif level <= 8:
                        rescue_needed = np.random.randint(7, 9)  # 7-8之间
                    else:
                        rescue_needed = np.random.randint(9, 11)  # 9-10之间
                    
                    # 新灾情点加入初始时间和时间步信息
                    self.disasters[(x, y)] = {
                        "level": level,
                        "rescue_needed": rescue_needed,
                        "time_step": self.current_time_step if current_time_step is None else current_time_step,  # 记录灾情点创建的时间步
                        "frozen_level": False,  # 初始状态为未冻结
                        "frozen_rescue": False,  # 初始状态为未冻结
                        "rescue_success": False,  # 初始状态为未救援成功
                        "show_red_x": 0  # 红叉显示计数器，0表示不显示
                    }
                    print(
                        f"🔴 新灾情点出现在 {x, y}，等级：{self.disasters[(x, y)]['level']}，需要救援：{self.disasters[(x, y)]['rescue_needed']}，时间步：{self.disasters[(x, y)]['time_step']}")

        # 自然减弱已有灾情（灾情会随时间自然减弱）
        for pos, disaster in list(self.disasters.items()):  # 使用list复制，避免在迭代中修改字典
            # 只跳过rescue_needed=0的灾情点，不再跳过level=0的点
            if disaster.get("frozen_rescue", False):
                continue

            if disaster["level"] > 0:
                # 统一自然减弱逻辑，无论训练模式还是部署模式都使用相同的减弱机制
                disaster["level"] -= np.random.randint(0, 2)  # 随机减弱0-1点
                disaster["level"] = max(0, disaster["level"])  # 确保不会为负

                # 如果自然减弱导致level降至0但rescue_needed>0，标记为救援失败
                if disaster["level"] <= 0 and disaster.get("rescue_needed", 0) > 0:
                    if not self.training_mode:  # 只在非训练模式下打印
                        print(f"⚠️ 灾情点 {pos} 自然减弱至level=0但仍需救援，标记为救援失败！")
                    disaster["show_red_x"] = 2  # 显示红叉
                    disaster["frozen_level"] = True  # 冻结level，防止进一步减弱
                    disaster["frozen_rescue"] = True  # 冻结救援状态
                    disaster["rescue_success"] = False  # 明确标记为救援失败
                    disaster["rescue_needed"] = 0  # 将rescue_needed设置为0，使其不再被计算为活跃灾难点
                    # 设置结束时间步
                    if current_time_step:
                        disaster["end_time_step"] = current_time_step
                    if not self.training_mode:  # 只在非训练模式下打印
                        print(f"⚪ 灾情点 {pos} 未能成功救援！")

    def update_disasters_silent(self, current_time_step=None):
        """
        无输出版本的灾难更新函数，用于训练过程中减少输出
        :param current_time_step: 当前的时间步，用于记录灾情点的创建时间
        """
        # 更新红叉计数器
        for pos, disaster in list(self.disasters.items()):
            # 如果红叉计数器大于0，则递减
            if disaster.get("show_red_x", 0) > 0:
                disaster["show_red_x"] -= 1

            # 记录前一个时间步的灾情等级，用于计算进度奖励
            if not disaster.get("frozen_level", False) and not disaster.get("frozen_rescue", False):
                disaster["previous_level"] = disaster["level"]

        # 获取当前时间步的实际灾情生成概率
        if current_time_step is not None:
            # 优先使用adjust_disaster_settings设置的概率
            if hasattr(self, 'disaster_gen_prob'):
                actual_spawn_rate = self.disaster_gen_prob
            else:
                actual_spawn_rate = config.get_actual_spawn_rate(current_time_step)
        else:
            # 如果没有提供时间步，使用基础概率
            if hasattr(self, 'disaster_gen_prob'):
                actual_spawn_rate = self.disaster_gen_prob
            else:
                actual_spawn_rate = config.get_config_param("disaster_spawn_rate")

        # 只在训练模式下启用数量上限控制
        if self.training_mode:
            # 统计当前活跃的灾难点数量（只计算需要救援的点）
            active_disaster_count = sum(1 for disaster in self.disasters.values() if disaster.get("rescue_needed", 0) > 0)
            max_disasters = self._get_disaster_limit(current_time_step)
            
            # 获取下限值（与adjust_disaster_settings保持一致）
            max_steps = getattr(self, '_max_steps', 300)
            phase1_end = int(max_steps * 2 / 3)
            phase2_end = int(max_steps * 5 / 6)
            
            if current_time_step is None or current_time_step < phase1_end:
                min_disasters = 20  # 初期阶段下限
            elif current_time_step < phase2_end:
                min_disasters = 5   # 中期阶段下限
            else:
                min_disasters = 1   # 后期阶段下限
            
            # 如果活跃灾难点不足，强制添加以达到下限
            if active_disaster_count < min_disasters:
                shortage = min_disasters - active_disaster_count
                added_count = 0
                max_add_attempts = shortage * 20  # 增加尝试次数
                
                for _ in range(max_add_attempts):
                    if added_count >= shortage:
                        break
                        
                    x, y = np.random.randint(0, self.GRID_SIZE, size=2)
                    if (x, y) not in self.disasters:
                        # 生成新的灾难点
                        level = np.random.randint(5, 11)  # 5-10之间
                        
                        # 统一rescue_needed的生成范围，与部署模式保持一致
                        if level <= 6:
                            rescue_needed = np.random.randint(5, 7)  # 5-6之间  
                        elif level <= 8:
                            rescue_needed = np.random.randint(7, 9)  # 7-8之间
                        else:
                            rescue_needed = np.random.randint(9, 11)  # 9-10之间
                        
                        # 新灾情点加入初始时间和时间步信息
                        self.disasters[(x, y)] = {
                            "level": level,
                            "rescue_needed": rescue_needed,
                            "time_step": self.current_time_step if current_time_step is None else current_time_step,
                            "frozen_level": False,
                            "frozen_rescue": False,
                            "rescue_success": False,
                            "show_red_x": 0
                        }
                        added_count += 1
            
            # 只有在未达到活跃灾难点上限时才生成新的灾情点
            elif active_disaster_count < max_disasters:
                # 增加灾难点生成数量，确保有足够的挑战性
                target_spawn_count = int(actual_spawn_rate * self.GRID_SIZE / 5)  # 从20改为5，增加生成数量
                if target_spawn_count < 2 and actual_spawn_rate > 0.1:
                    target_spawn_count = 2  # 确保至少尝试生成2个
                
                # 限制生成数量不超过上限
                max_new_disasters = max_disasters - active_disaster_count
                target_spawn_count = min(target_spawn_count, max_new_disasters)
                
                # 随机选择位置生成灾难点
                new_disasters_count = 0
                attempts = 0
                max_attempts = target_spawn_count * 10  # 最多尝试10倍的次数
                
                while new_disasters_count < target_spawn_count and attempts < max_attempts:
                    x, y = np.random.randint(0, self.GRID_SIZE, size=2)
                    attempts += 1
                    
                    # 如果位置已被占用，跳过
                    if (x, y) in self.disasters:
                        continue
                    
                    # 生成新的灾难点
                    level = np.random.randint(5, 11)  # 5-10之间
                    
                    # 统一rescue_needed的生成范围，与部署模式保持一致
                    if level <= 6:
                        rescue_needed = np.random.randint(5, 7)  # 5-6之间  
                    elif level <= 8:
                        rescue_needed = np.random.randint(7, 9)  # 7-8之间
                    else:
                        rescue_needed = np.random.randint(9, 11)  # 9-10之间
                    
                    # 新灾情点加入初始时间和时间步信息
                    self.disasters[(x, y)] = {
                        "level": level,
                        "rescue_needed": rescue_needed,
                        "time_step": self.current_time_step if current_time_step is None else current_time_step,  # 记录灾情点创建的时间步
                        "frozen_level": False,  # 初始状态为未冻结
                        "frozen_rescue": False,  # 初始状态为未冻结
                        "rescue_success": False,  # 初始状态为未救援成功
                        "show_red_x": 0  # 红叉显示计数器，0表示不显示
                    }
                    new_disasters_count += 1
        else:
            # 非训练模式，不启用数量上限控制
            # 随机生成新的灾情点
            for _ in range(int(actual_spawn_rate * self.GRID_SIZE)):
                x, y = np.random.randint(0, self.GRID_SIZE, size=2)
                if (x, y) not in self.disasters:
                    # 先生成level，范围5-10
                    level = np.random.randint(5, 11)  # 注意上限改为11，使范围包含10
                    
                    # 统一rescue_needed的生成范围，与部署模式保持一致
                    if level <= 6:
                        rescue_needed = np.random.randint(5, 7)  # 5-6之间  
                    elif level <= 8:
                        rescue_needed = np.random.randint(7, 9)  # 7-8之间
                    else:
                        rescue_needed = np.random.randint(9, 11)  # 9-10之间
                    
                    # 新灾情点加入初始时间和时间步信息
                    self.disasters[(x, y)] = {
                        "level": level,
                        "rescue_needed": rescue_needed,
                        "time_step": self.current_time_step if current_time_step is None else current_time_step,  # 记录灾情点创建的时间步
                        "frozen_level": False,  # 初始状态为未冻结
                        "frozen_rescue": False,  # 初始状态为未冻结
                        "rescue_success": False,  # 初始状态为未救援成功
                        "show_red_x": 0  # 红叉显示计数器，0表示不显示
                    }
                    print(
                        f"🔴 新灾情点出现在 {x, y}，等级：{self.disasters[(x, y)]['level']}，需要救援：{self.disasters[(x, y)]['rescue_needed']}，时间步：{self.disasters[(x, y)]['time_step']}")

        # 自然减弱已有灾情（与update_disasters方法保持一致）
        for pos, disaster in list(self.disasters.items()):  # 使用list复制，避免在迭代中修改字典
            # 只跳过rescue_needed=0的灾情点，不再跳过level=0的点
            if disaster.get("frozen_rescue", False):
                continue

            if disaster["level"] > 0:
                # 统一自然减弱逻辑，无论训练模式还是部署模式都使用相同的减弱机制
                disaster["level"] -= np.random.randint(0, 2)  # 随机减弱0-1点
                disaster["level"] = max(0, disaster["level"])  # 确保不会为负

                # 如果自然减弱导致level降至0但rescue_needed>0，标记为救援失败
                if disaster["level"] <= 0 and disaster.get("rescue_needed", 0) > 0:
                    disaster["show_red_x"] = 2  # 显示红叉
                    disaster["frozen_level"] = True  # 冻结level，防止进一步减弱
                    disaster["frozen_rescue"] = True  # 冻结救援状态
                    disaster["rescue_success"] = False  # 明确标记为救援失败
                    disaster["rescue_needed"] = 0  # 将rescue_needed设置为0，使其不再被计算为活跃灾难点
                    # 设置结束时间步
                    if current_time_step:
                        disaster["end_time_step"] = current_time_step

    def _get_disaster_limit(self, current_time_step=None, max_steps=None):
        """
        根据当前时间步获取灾难数量上限
        这个方法与rl_util.py中的逻辑保持一致
        """
        if current_time_step is None:
            # 如果没有时间步信息，返回默认上限
            return 50
        
        # 获取最大步数 - 优先使用传入的参数，否则使用硬编码的300步
        if max_steps is None:
            max_steps = getattr(self, '_max_steps', 300)  # 硬编码为300步
        
        # 计算阶段边界（与adjust_disaster_settings保持一致）
        phase1_end = int(max_steps * 2 / 3)  # 初期结束
        phase2_end = int(max_steps * 5 / 6)  # 中期结束
        
        # 根据训练阶段返回不同的灾难数量上限，与adjust_disaster_settings保持一致
        if current_time_step < phase1_end:  # 初期阶段
            return 50  # 对应adjust_disaster_settings中的max_disasters = 50
        elif current_time_step < phase2_end:  # 中期阶段
            return 20  # 对应adjust_disaster_settings中的max_disasters = 20
        else:  # 后期阶段
            return 5   # 对应adjust_disaster_settings中的max_disasters = 5

    def get_state_for_rescuer(self, rescuer_idx):
        """
        获取指定救援人员的状态表示
        这个方法提供了与RescueEnvironment.get_state_for_rescuer兼容的接口
        """
        # 这里我们需要导入MARLController来构建状态
        try:
            from src.rl.marl_rescue import MARLController
            
            # 创建一个临时的MARLController以获取状态表示
            controller = MARLController(
                env_or_grid_size=self.GRID_SIZE, 
                num_rescuers=len(self.rescuers)
            )
            return controller.build_state(rescuer_idx, self.rescuers, self.disasters)
        except Exception as e:
            print(f"构建状态时出错: {e}")
            # 返回一个零张量作为默认状态
            import torch
            import numpy as np
            state_dim = 4 + self.GRID_SIZE * self.GRID_SIZE * 5
            return torch.zeros((1, state_dim))
    
    def is_episode_done(self):
        """
        检查当前模拟是否结束
        """
        # 检查是否所有灾难都已解决
        all_resolved = True
        for disaster in self.disasters.values():
            if disaster["rescue_needed"] > 0:
                all_resolved = False
                break
        
        # 检查是否达到最大时间步 - 使用动态的max_steps而不是硬编码的SIMULATION_TIME
        max_steps = getattr(self, '_max_steps', None)
        if max_steps is not None:
            time_limit_reached = self.current_time_step >= max_steps - 1
        else:
            # 如果没有设置_max_steps，则使用config中的默认值
            from src.core import config
            time_limit_reached = self.current_time_step >= config.SIMULATION_TIME - 1
        
        return all_resolved or time_limit_reached
    
    def step(self, rescuer_idx, action):
        """
        执行单个救援人员的动作并返回结果
        
        参数:
            rescuer_idx: 救援人员索引
            action: 动作索引
        
        返回:
            next_state: 下一个状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 确保救援者索引有效
        if rescuer_idx >= len(self.rescuers):
            print(f"警告: 救援者索引 {rescuer_idx} 超出了范围 {len(self.rescuers)}")
            # 返回零状态、负奖励和结束标志
            return self.get_state_for_rescuer(0), -1, True, {"success": False, "response_time": 0}
        
        # 记录旧状态和灾情
        old_state = self.rescuers[rescuer_idx].copy() if rescuer_idx < len(self.rescuers) else {}
        old_disasters = {pos: disaster.copy() for pos, disaster in self.disasters.items()}
        
        # 将动作转换为目标位置
        grid_size = self.GRID_SIZE
        target_pos = None
        
        if action > 0:  # 非0动作表示前往某个位置
            action_idx = action - 1
            x = action_idx // grid_size
            y = action_idx % grid_size
            target_pos = (x, y)
            
            # 如果目标位置有灾情，则分配任务
            if target_pos in self.disasters:
                self.rescuers[rescuer_idx]["target"] = target_pos
            else:
                # 如果目标位置没有灾情，则清除当前任务
                if "target" in self.rescuers[rescuer_idx]:
                    del self.rescuers[rescuer_idx]["target"]
        
        # 执行救援
        from src.core.rescue_execution import execute_rescue
        execute_rescue(self.rescuers, self.disasters, self.GRID_SIZE, current_time_step=self.current_time_step)
        
        # 使用rl_util中的奖励计算函数
        try:
            from src.rl.rl_util import calculate_reward
            reward, reward_info = calculate_reward(self, rescuer_idx, old_state, old_disasters)
            
            # 打印调试信息
            if self.current_time_step % 10 == 0 and rescuer_idx == 0:
                print(f"[调试] 计算奖励: {reward:.4f}, 奖励明细: {reward_info}")
        except Exception as e:
            print(f"[错误] 计算奖励时出错: {e}")
            # 发生错误时，使用默认奖励
            reward = -0.01
            reward_info = {"time_penalty": -0.01}
        
        # 确定成功标志和响应时间
        success = False
        response_time = 0
        
        # 如果有目标且目标是灾情点且灾情已解决或减轻，则计算响应时间
        if "target" in self.rescuers[rescuer_idx]:
            target = self.rescuers[rescuer_idx]["target"]
            if target in old_disasters and target in self.disasters:
                old_disaster = old_disasters[target]
                current_disaster = self.disasters[target]
                
                # 如果救援取得了进展
                if old_disaster["rescue_needed"] > current_disaster["rescue_needed"]:
                    success = True
                    
                    # 计算响应时间 - 使用时间步而不是实际时间
                    if "time_step" in current_disaster:
                        response_time = self.current_time_step - current_disaster["time_step"]
                        # 确保响应时间为正数
                        if response_time <= 0:
                            print(f"警告：检测到负响应时间，current_time_step={self.current_time_step}, disaster_time_step={current_disaster['time_step']}")
                            response_time = 1  # 确保至少为1
        
        # 获取新状态
        next_state = self.get_state_for_rescuer(rescuer_idx)
        
        # 判断是否结束
        done = self.is_episode_done()
        
        # 返回结果
        info = {
            "success": success,
            "response_time": response_time
        }
        
        return next_state, reward, done, info

    def set_rescuers(self, rescuers_data):
        """设置救援人员列表，用于在部署时加载训练好的救援人员参数"""
        if rescuers_data:
            self.rescuers = rescuers_data
            self.num_rescuers = len(rescuers_data)
            print(f"环境已更新为使用加载的救援人员数据，数量: {self.num_rescuers}")
        return self.rescuers


if __name__ == "__main__":
    env = Environment(GRID_SIZE, NUM_RESCUERS)  # Create environment instance
    print("City Map:")  # 输出城市地图
    print(env.disasters)  # 输出当前灾情信息

    print("\nDisaster Points:")  # 输出灾情点信息
    for key, value in env.disasters.items():
        print(f"Location {key}, Level: {value['level']}, Rescue Needed: {value['rescue_needed']}")  # 输出每个灾情点的具体信息

    print("\nRescuer Information:")  # 输出救援人员信息
    for rescuer in env.rescuers:
        print(
            f"ID {rescuer['id']}, Position: {rescuer['position']}, Speed: {rescuer['speed']}, Capacity: {rescuer['capacity']}")  # 输出救援人员编号、位置、速度和救援能力
