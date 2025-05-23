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
    TIME_PENALTY = 1          # 时间惩罚
    
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
    # 设置环境的最大步数属性，供_get_disaster_limit使用
    env._max_steps = max_steps
    
    # 获取当前活跃的灾难数量（只计算需要救援的点）
    if hasattr(env, "disasters"):
        current_disasters = sum(1 for disaster in env.disasters.values() if disaster.get("rescue_needed", 0) > 0)
    elif hasattr(env, "disaster_locations"):
        current_disasters = len(env.disaster_locations)
    else:
        current_disasters = 0
    
    # 计算阶段边界（使用整数避免浮点数精度问题）
    phase1_end = int(max_steps * 2 / 3)  # 初期结束
    phase2_end = int(max_steps * 5 / 6)  # 中期结束
    
    # 根据训练阶段调整灾难生成概率和灾难点数量范围
    if step < phase1_end:  # 初期阶段
        # 灾难初期：高频率灾难点生成，确保有足够的灾难点进行训练
        base_prob = 0.5
        min_disasters = 20
        max_disasters = 50
        phase = "初期阶段"
    elif step < phase2_end:  # 中期阶段
        # 灾难中期：中等频率灾难点生成，灾难点数量适中
        base_prob = 0.3
        min_disasters = 5
        max_disasters = 20
        phase = "中期阶段"
    else:  # 后期阶段
        # 灾难后期：低频率灾难点生成，灾难点数量减少
        base_prob = 0.1
        min_disasters = 1
        max_disasters = 5
        phase = "后期阶段"
    
    # 动态调整生成概率：如果活跃灾难点数量低于下限，提高生成概率
    if current_disasters < min_disasters:
        # 根据缺口大小动态提高生成概率
        shortage = min_disasters - current_disasters
        boost_factor = min(3.0, 1.0 + shortage * 0.2)  # 最多提高到3倍
        env.disaster_gen_prob = base_prob * boost_factor
        if verbose:
            print(f"🔥 灾难点不足，提高生成概率：{base_prob:.1f} -> {env.disaster_gen_prob:.1f} (缺口: {shortage})")
    else:
        env.disaster_gen_prob = base_prob
    
    # 打印当前的灾难管理策略（每50步显示一次）
    if verbose or step % 50 == 0:
        print(f"\033[33m当前{phase}：灾难生成概率={env.disaster_gen_prob:.1f}, 活跃灾难点范围={min_disasters}-{max_disasters}个，当前有{current_disasters}个活跃灾难点\033[0m")
    
    # 强制补充逻辑：当活跃灾难点数量不足最小值时，直接添加新的灾难点
    if current_disasters < min_disasters and hasattr(env, "disasters"):
        shortage = min_disasters - current_disasters
        added_count = _force_add_disasters(env, shortage, verbose=verbose)
        if added_count > 0:
            print(f"🚨 强制补充：活跃灾难点不足，已添加{added_count}个新灾难点（目标缺口: {shortage}）")
    
    # 当活跃灾难点数量超过最大值时，智能移除灾难点（保护正在被救援的点）
    if current_disasters > max_disasters and hasattr(env, "disasters"):
        # 使用改进的智能减少方法，保护正在被救援的灾难点
        _smart_reduce_disasters(env, max_disasters, verbose=False)
        print(f"🔄 阶段变化：活跃灾难点从{current_disasters}个减少到上限{max_disasters}个")

def _smart_reduce_disasters(env, target_count, verbose=False):
    """
    智能减少活跃灾难点数量，优先保护正在被救援的灾难点
    
    参数:
        env: 环境对象
        target_count: 目标活跃灾难点数量上限
        verbose: 是否输出详细信息
    """
    # 检查环境是否有灾难点属性
    if not hasattr(env, "disasters"):
        return False
    
    # 获取当前活跃灾难点数量（只计算需要救援的点）
    current_active_count = sum(1 for disaster in env.disasters.values() if disaster.get("rescue_needed", 0) > 0)
    
    # 如果当前活跃灾难点数量小于等于目标值，不需要处理
    if current_active_count <= target_count:
        return True
    
    # 需要移除的活跃灾难点数量
    to_remove = current_active_count - target_count
    
    # 获取正在被救援的灾难点位置（被救援人员作为目标的点）
    protected_positions = set()
    if hasattr(env, "rescuers"):
        for rescuer in env.rescuers:
            if "target" in rescuer and rescuer["target"] in env.disasters:
                protected_positions.add(rescuer["target"])
    
    # 获取可以移除的活跃灾难点（不包括被保护的点，只考虑需要救援的点）
    removable_positions = []
    for pos, disaster in env.disasters.items():
        # 只考虑活跃的灾难点（需要救援的点）
        if disaster.get("rescue_needed", 0) <= 0:
            continue
        # 跳过正在被救援的灾难点
        if pos in protected_positions:
            continue
        # 跳过已冻结的灾难点（已完成或失败的救援）
        if disaster.get("frozen_level", False) or disaster.get("frozen_rescue", False):
            continue
        # 可以移除的点
        removable_positions.append(pos)
    
    # 计算实际可移除的数量
    actual_removable = min(to_remove, len(removable_positions))
    
    if actual_removable == 0:
        if verbose:
            print(f"⚠️ 无法移除任何活跃灾难点：所有{current_active_count}个活跃灾难点都被保护或已冻结")
        return False
    
    # 如果可移除的数量不足以达到目标，发出警告
    if actual_removable < to_remove:
        if verbose:
            print(f"⚠️ 只能移除{actual_removable}个活跃灾难点（目标需要移除{to_remove}个），因为{len(protected_positions)}个正在被救援，其余已冻结")
    
    # 随机选择要移除的灾难点
    positions_to_remove = random.sample(removable_positions, actual_removable)
    
    # 移除选定的灾难点
    removed = 0
    for pos in positions_to_remove:
        try:
            del env.disasters[pos]
            removed += 1
        except Exception as e:
            if verbose:
                print(f"移除灾难点{pos}时出错: {e}")
    
    if verbose:
        # 重新计算活跃灾难点数量
        new_active_count = sum(1 for disaster in env.disasters.values() if disaster.get("rescue_needed", 0) > 0)
        print(f"成功移除{removed}个灾难点，当前活跃灾难点数量：{new_active_count}（保护了{len(protected_positions)}个正在被救援的点）")
    
    return removed > 0

def _force_add_disasters(env, count, verbose=False):
    """
    强制添加指定数量的新灾难点
    
    参数:
        env: 环境对象
        count: 需要添加的灾难点数量
        verbose: 是否输出详细信息
        
    返回:
        实际添加的灾难点数量
    """
    import random
    from src.core import config
    
    if not hasattr(env, "disasters") or not hasattr(env, "GRID_SIZE"):
        return 0
    
    added_count = 0
    max_attempts = count * 10  # 防止无限循环
    attempts = 0
    
    while added_count < count and attempts < max_attempts:
        attempts += 1
        
        # 随机选择一个空的位置
        x = random.randint(0, env.GRID_SIZE - 1)
        y = random.randint(0, env.GRID_SIZE - 1)
        pos = (x, y)
        
        # 如果该位置已经有灾难点，跳过
        if pos in env.disasters:
            continue
        
        # 创建新的灾难点
        disaster_level = random.uniform(1, config.CRITICAL_DISASTER_THRESHOLD)
        rescue_needed = random.randint(1, config.MAX_RESCUE_CAPACITY)
        
        # 添加灾难点
        env.disasters[pos] = {
            "level": disaster_level,
            "rescue_needed": rescue_needed,
            "time_step": getattr(env, 'current_time_step', 0),  # 记录创建时间
            "rescue_success": False,  # 初始状态为未成功
            "frozen_level": False,    # 未冻结等级
            "frozen_rescue": False    # 未冻结救援状态
        }
        
        added_count += 1
        
        if verbose:
            print(f"  ➕ 在位置({x}, {y})添加新灾难点：等级={disaster_level:.1f}, 需救援={rescue_needed}")
    
    if verbose and added_count < count:
        print(f"⚠️ 只成功添加了{added_count}/{count}个灾难点（尝试{attempts}次后停止）")
    
    return added_count

def _force_reduce_disasters(env, target_count, verbose=False):
    """
    保留原有的强制减少函数以向后兼容，但现在调用智能减少函数
    """
    return _smart_reduce_disasters(env, target_count, verbose) 