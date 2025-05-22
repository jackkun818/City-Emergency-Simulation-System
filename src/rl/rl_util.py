"""
强化学习工具函数

包含奖励计算和其他辅助功能
"""

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