"""
强化学习工具函数

包含奖励计算和其他辅助功能
"""
import numpy as np
import random
import time

class Colors:
    HEADER = '\033[95m'     # 粉色
    BLUE = '\033[94m'       # 蓝色
    CYAN = '\033[96m'       # 青色
    GREEN = '\033[92m'      # 绿色
    YELLOW = '\033[93m'     # 黄色
    RED = '\033[91m'        # 红色
    ENDC = '\033[0m'        # 结束颜色
    BOLD = '\033[1m'        # 粗体
    UNDERLINE = '\033[4m'   # 下划线

def calculate_reward(env, rescuer_idx, old_state, old_disasters):
    """
    计算单个救援人员的奖励
    
    参数:
    - env: 环境对象，必须包含rescuers和disasters属性
    - rescuer_idx: 救援人员索引
    - old_state: 动作执行前的救援人员状态
    - old_disasters: 动作执行前的灾情状态
    
    返回:
    - reward: 计算得到的奖励值
    - reward_info: 奖励明细
    """
    # 奖励系数 - 统一在一处定义
    COMPLETION_REWARD = 10.0    # 完成救援奖励
    PRIORITY_FACTOR = 0.1       # 优先级因子 (乘以灾情等级)
    COORDINATION_REWARD = 2.0   # 协调奖励
    PROGRESS_FACTOR = 1.0       # 进度奖励因子 (乘以救援进度)
    TIME_PENALTY = 0.1          # 时间惩罚
    
    # 确保rescuer_idx在有效范围内
    if rescuer_idx >= len(env.rescuers):
        return 0, {"completion_reward": 0, "priority_reward": 0, 
                 "coordination_reward": 0, "progress_reward": 0, "time_penalty": 0}
        
    rescuer = env.rescuers[rescuer_idx]
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
        if target in env.disasters:
            current_disaster = env.disasters[target]
            old_disaster = old_disasters.get(target, None)
            
            # 奖励1: 根据灾情等级给予奖励
            priority_reward = current_disaster["level"] * PRIORITY_FACTOR
            reward_info["priority_reward"] = priority_reward
            reward += priority_reward
            
            # 奖励2: 协调奖励 - 检查是否有其他救援人员前往同一目标
            other_targeting = False
            for i, other_rescuer in enumerate(env.rescuers):
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

def adjust_disaster_settings(env, step, max_steps, verbose=False):
    """根据训练进度动态调整灾难设置"""
    # 获取当前的灾难数量
    current_disasters = len(env.disasters) if hasattr(env, "disasters") else len(env.disaster_locations) if hasattr(env, "disaster_locations") else 0
    
    # 根据训练阶段调整灾难生成概率和灾难点数量范围
    if step < max_steps / 3:  # 初期阶段（前1/3训练）
        # 灾难初期：高频率灾难点生成，确保有足够的灾难点进行训练
        env.disaster_gen_prob = 0.5 if hasattr(env, "disaster_gen_prob") else 0.5
        min_disasters = 20
        max_disasters = 50
        phase = "初期阶段"
    elif step < 2 * max_steps / 3:  # 中期阶段（中间1/3训练）
        # 灾难中期：中等频率灾难点生成，灾难点数量适中
        env.disaster_gen_prob = 0.3 if hasattr(env, "disaster_gen_prob") else 0.3
        min_disasters = 5
        max_disasters = 20
        phase = "中期阶段"
    else:  # 后期阶段（最后1/3训练）
        # 灾难后期：低频率灾难点生成，灾难点数量减少
        env.disaster_gen_prob = 0.1 if hasattr(env, "disaster_gen_prob") else 0.1
        min_disasters = 1
        max_disasters = 5
        phase = "后期阶段"
    
    # 打印当前的灾难管理策略（每50步显示一次）
    if verbose or step % 50 == 0:
        print(f"\033[33m当前{phase}：灾难生成概率={env.disaster_gen_prob if hasattr(env, 'disaster_gen_prob') else 0.5:.1f}, 灾难点范围={min_disasters}-{max_disasters}个，当前有{current_disasters}个灾难点\033[0m")
    
    # 使用强制方法处理灾难点数量
    # 当灾难点数量少于最小值时，添加灾难点
    if current_disasters < min_disasters and hasattr(env, "disasters"):
        to_add = min_disasters - current_disasters
        
        # 尝试添加灾难点
        for _ in range(to_add):
            # 找一个未被占用的位置
            grid_size = env.GRID_SIZE if hasattr(env, "GRID_SIZE") else env.grid_size if hasattr(env, "grid_size") else 10
            max_attempts = 10
            
            for _ in range(max_attempts):
                x, y = np.random.randint(0, grid_size, size=2)
                if (x, y) not in env.disasters:
                    # 生成一个新的灾难点
                    level = np.random.randint(5, 11)
                    
                    if level <= 6:
                        rescue_needed = np.random.randint(5, 6)
                    elif level <= 8:
                        rescue_needed = np.random.randint(7, 8)
                    else:
                        rescue_needed = np.random.randint(9, 10)
                    
                    # 添加新灾难点
                    env.disasters[(x, y)] = {
                        "level": level,
                        "rescue_needed": rescue_needed,
                        "start_time": time.time(),
                        "time_step": env.current_time_step if hasattr(env, "current_time_step") else 0,
                        "frozen_level": False,
                        "frozen_rescue": False,
                        "rescue_success": False,
                        "show_red_x": 0
                    }
                    break
    
    # 当灾难点数量超过最大值时，移除灾难点
    if current_disasters > max_disasters and hasattr(env, "disasters"):
        # 直接使用强制方法确保灾难点数量不超过最大值
        _force_reduce_disasters(env, max_disasters, verbose=False)

def _force_reduce_disasters(env, target_count, verbose=False):
    """
    直接强制管理灾难点数量，确保环境中的灾难点数量不超过目标值
    
    参数:
        env: 环境对象
        target_count: 目标灾难点数量上限
        verbose: 是否输出详细信息
    """
    # 检查环境是否有灾难点属性
    if not hasattr(env, "disasters"):
        return False
    
    # 获取当前灾难点数量
    current_count = len(env.disasters)
    
    # 如果当前灾难点数量小于等于目标值，不需要处理
    if current_count <= target_count:
        return True
    
    # 需要移除的灾难点数量
    to_remove = current_count - target_count
    
    # 获取所有灾难点位置
    disaster_positions = list(env.disasters.keys())
    
    # 随机选择要移除的灾难点
    positions_to_remove = random.sample(disaster_positions, to_remove)
    
    # 移除选定的灾难点
    removed = 0
    for pos in positions_to_remove:
        try:
            del env.disasters[pos]
            removed += 1
        except Exception as e:
            pass
    
    return removed > 0 