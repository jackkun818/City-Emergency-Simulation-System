#!/usr/bin/env python3
"""
多智能体强化学习(MARL)训练脚本

用法:
  python train.py --episodes 1000 --steps 100 --save-freq 5

参数:
  --episodes   训练轮数 (默认: 50)
  --steps      每轮训练步数 (默认: 300)
  --save-freq  保存频率，每多少轮保存一次模型 (默认: 5)
"""

import sys
import os
import io
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import contextlib
import time
import random
import argparse
import json
import pickle
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
# print(project_root)
# raise Exception("test")
# 导入自定义模块
from src.core.environment import Environment
from src.rl.marl_rescue import train_marl, MARLController
from src.core import config
# from src.utils.colors import Colors

# 导入可视化模块
from src.visualization.visualization import visualize

# 定义存储目录

SAVE_DIR = os.path.join(project_root, 'train_visualization_save')
os.makedirs(SAVE_DIR, exist_ok=True)

# 定义ANSI颜色代码
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练多智能体强化学习模型')
    parser.add_argument('--episodes', type=int, default=50, help='训练轮数 ')
    parser.add_argument('--steps', type=int, default=300, help='每轮训练步数 ')
    parser.add_argument('--save-freq', type=int, default=5, help='保存频率，每多少轮保存一次模型 (默认: 5)')
    return parser.parse_args()


def visualize_training_results(rewards, success_rates, response_times, save_path="./training_results"):
    """可视化训练结果"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 绘制平均奖励
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('平均奖励变化')
    plt.xlabel('训练轮数')
    plt.ylabel('平均奖励')
    plt.grid(True)
    plt.savefig(f"{save_path}/rewards.png")
    plt.close()
    
    # 绘制救援成功率
    plt.figure(figsize=(10, 6))
    plt.plot(success_rates)
    plt.title('救援成功率变化')
    plt.xlabel('训练轮数')
    plt.ylabel('救援成功率')
    plt.grid(True)
    plt.savefig(f"{save_path}/success_rates.png")
    plt.close()
    
    # 绘制平均响应时间
    plt.figure(figsize=(10, 6))
    plt.plot(response_times)
    plt.title('平均响应时间变化')
    plt.xlabel('训练轮数')
    plt.ylabel('平均响应时间')
    plt.grid(True)
    plt.savefig(f"{save_path}/response_times.png")
    plt.close()
    
    # 保存原始数据
    np.save(f"{save_path}/rewards.npy", np.array(rewards))
    np.save(f"{save_path}/success_rates.npy", np.array(success_rates))
    np.save(f"{save_path}/response_times.npy", np.array(response_times))
    
    print(f"训练结果可视化已保存到: {save_path}")


def _remove_disasters_from_locations(env, num_to_remove, verbose=False):
    """从disaster_locations移除灾难点"""
    # 确保不会尝试移除超过实际灾难点数量
    num_to_remove = min(num_to_remove, len(env.disaster_locations))
    
    if num_to_remove <= 0:
        return
    
    # 将集合转换为列表以便随机抽样
    disaster_list = list(env.disaster_locations)
    
    # 随机选择要移除的灾难点
    disasters_to_remove = random.sample(disaster_list, num_to_remove)
    
    # 移除选中的灾难点
    removed_count = 0
    for disaster in disasters_to_remove:
        try:
            env.disaster_locations.remove(disaster)
            removed_count += 1
        except Exception as e:
            if verbose:
                print(f"移除灾难点时出错: {e}")
    
    if verbose:
        print(f"从locations移除了{removed_count}个灾难点，当前灾难点数量：{len(env.disaster_locations)}")
    
    # 强制确保灾难点数量不超过设定的最大值（紧急修复）
    if hasattr(env, "_force_reduce_disasters") and callable(env._force_reduce_disasters):
        current_disasters = len(env.disaster_locations)
        max_disasters = _get_max_disasters_for_phase(env, env.current_time_step)
        if current_disasters > max_disasters:
            extra = current_disasters - max_disasters
            if verbose:
                print(f"强制移除额外的{extra}个灾难点")
            env._force_reduce_disasters(extra)


def _get_max_disasters_for_phase(env, step):
    """根据当前步骤返回最大灾难点数量"""
    max_steps = env.SIMULATION_TIME if hasattr(env, "SIMULATION_TIME") else 500
    
    if step < max_steps / 3:  # 初期阶段
        return 50
    elif step < 2 * max_steps / 3:  # 中期阶段
        return 20
    else:  # 后期阶段
        return 5


# 添加一个紧急修复函数用于强制管理灾难点数量
def _emergency_fix_disaster_count(env, max_disasters, verbose=False):
    """紧急修复：强制减少灾难点数量"""
    if not hasattr(env, "disasters") and not hasattr(env, "disaster_locations"):
        if verbose:
            print("无法应用紧急修复：环境没有灾难点属性")
        return
    
    # 确定当前灾难点数量
    current_disasters = len(env.disasters) if hasattr(env, "disasters") else len(env.disaster_locations)
    
    # 如果灾难点数量正常，不需要修复
    if current_disasters <= max_disasters:
        return
    
    # 需要移除的灾难点数量
    to_remove = current_disasters - max_disasters
    
    if verbose:
        print(f"紧急修复：当前有{current_disasters}个灾难点，需要减少到{max_disasters}个")
    
    # 根据环境接口选择合适的方法进行灾难点管理
    if hasattr(env, "disasters"):
        _remove_disasters_from_env(env, to_remove, verbose=verbose)
    elif hasattr(env, "disaster_locations"):
        _remove_disasters_from_locations(env, to_remove, verbose=verbose)
    
    # 验证修复结果
    new_count = len(env.disasters) if hasattr(env, "disasters") else len(env.disaster_locations)
    if verbose:
        print(f"修复后灾难点数量：{new_count}，应为不超过{max_disasters}个")


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


# def adjust_disaster_settings(env, step, max_steps, verbose=False):
#     """根据训练进度动态调整灾难设置"""
#     # 获取当前的灾难数量
#     current_disasters = len(env.disasters) if hasattr(env, "disasters") else len(env.disaster_locations) if hasattr(env, "disaster_locations") else 0
    
#     # 根据训练阶段调整灾难生成概率和灾难点数量范围
#     if step < max_steps / 3:  # 初期阶段（前1/3训练）
#         # 灾难初期：高频率灾难点生成，确保有足够的灾难点进行训练
#         env.disaster_gen_prob = 0.5 if hasattr(env, "disaster_gen_prob") else 0.5
#         min_disasters = 20
#         max_disasters = 50
#         phase = "初期阶段"
#     elif step < 2 * max_steps / 3:  # 中期阶段（中间1/3训练）
#         # 灾难中期：中等频率灾难点生成，灾难点数量适中
#         env.disaster_gen_prob = 0.3 if hasattr(env, "disaster_gen_prob") else 0.3
#         min_disasters = 5
#         max_disasters = 20
#         phase = "中期阶段"
#     else:  # 后期阶段（最后1/3训练）
#         # 灾难后期：低频率灾难点生成，灾难点数量减少
#         env.disaster_gen_prob = 0.1 if hasattr(env, "disaster_gen_prob") else 0.1
#         min_disasters = 1
#         max_disasters = 5
#         phase = "后期阶段"
    
#     # 打印当前的灾难管理策略（每50步显示一次）
#     if verbose or step % 50 == 0:
#         print(f"\033[33m当前{phase}：灾难生成概率={env.disaster_gen_prob if hasattr(env, 'disaster_gen_prob') else 0.5:.1f}, 灾难点范围={min_disasters}-{max_disasters}个，当前有{current_disasters}个灾难点\033[0m")
    
#     # 使用强制方法处理灾难点数量
#     # 当灾难点数量少于最小值时，添加灾难点
#     if current_disasters < min_disasters and hasattr(env, "disasters"):
#         to_add = min_disasters - current_disasters
        
#         # 尝试添加灾难点
#         for _ in range(to_add):
#             # 找一个未被占用的位置
#             grid_size = env.GRID_SIZE if hasattr(env, "GRID_SIZE") else env.grid_size if hasattr(env, "grid_size") else 10
#             max_attempts = 10
            
#             for _ in range(max_attempts):
#                 x, y = np.random.randint(0, grid_size, size=2)
#                 if (x, y) not in env.disasters:
#                     # 生成一个新的灾难点
#                     level = np.random.randint(5, 11)
                    
#                     if level <= 6:
#                         rescue_needed = np.random.randint(5, 6)
#                     elif level <= 8:
#                         rescue_needed = np.random.randint(7, 8)
#                     else:
#                         rescue_needed = np.random.randint(9, 10)
                    
#                     # 添加新灾难点
#                     env.disasters[(x, y)] = {
#                         "level": level,
#                         "rescue_needed": rescue_needed,
#                         "start_time": time.time(),
#                         "time_step": env.current_time_step if hasattr(env, "current_time_step") else 0,
#                         "frozen_level": False,
#                         "frozen_rescue": False,
#                         "rescue_success": False,
#                         "show_red_x": 0
#                     }
#                     break
    
#     # 当灾难点数量超过最大值时，移除灾难点
#     if current_disasters > max_disasters and hasattr(env, "disasters"):
#         # 直接使用强制方法确保灾难点数量不超过最大值
#         _force_reduce_disasters(env, max_disasters, verbose=False)


def _remove_disasters_from_env(env, num_to_remove, verbose=False):
    """根据环境接口智能移除灾难点，保护正在被救援的点"""
    if hasattr(env, "disasters"):
        # 获取正在被救援的灾难点位置（被救援人员作为目标的点）
        protected_positions = set()
        if hasattr(env, "rescuers"):
            for rescuer in env.rescuers:
                if "target" in rescuer and rescuer["target"] in env.disasters:
                    protected_positions.add(rescuer["target"])
        
        # 获取可移除的灾难点（不包括正在被救援的点和已冻结的点）
        removable_disasters = []
        
        for pos, disaster in env.disasters.items():
            # 跳过正在被救援的灾难点
            if pos in protected_positions:
                continue
            # 跳过已冻结的灾难点（已完成或失败的救援）
            if disaster.get("frozen_level", False) or disaster.get("frozen_rescue", False):
                continue
            # 可以移除的点
            removable_disasters.append(pos)
        
        # 确保不会尝试移除超过实际可移除灾难点数量
        actual_removable = min(num_to_remove, len(removable_disasters))
        
        if actual_removable == 0:
            if verbose:
                print(f"⚠️ 无法移除任何灾难点：{len(protected_positions)}个正在被救援，其余已冻结")
            return
        
        # 如果可移除的数量不足以达到目标，发出警告
        if actual_removable < num_to_remove:
            if verbose:
                print(f"⚠️ 只能移除{actual_removable}个灾难点（目标需要移除{num_to_remove}个），因为{len(protected_positions)}个正在被救援")
        
        # 随机选择要移除的灾难点
        if removable_disasters:
            disasters_to_remove = random.sample(removable_disasters, actual_removable)
            
            # 移除选中的灾难点
            for disaster in disasters_to_remove:
                del env.disasters[disaster]
            
            if verbose:
                print(f"成功移除{actual_removable}个灾难点，当前灾难点数量：{len(env.disasters)}（保护了{len(protected_positions)}个正在被救援的点）")


def _add_disasters_to_locations(env, num_to_add, verbose=False):
    """向disaster_locations添加灾难点"""
    grid_size = env.grid_size if hasattr(env, "grid_size") else 10
    for _ in range(num_to_add):
        # 尝试添加新的灾难点
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            # 随机生成新的灾难点位置
            new_loc = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            
            # 检查是否与现有灾难点重叠
            if new_loc not in env.disaster_locations:
                # 检查是否与智能体位置重叠
                overlap_with_agent = False
                if hasattr(env, "agents"):
                    for agent in env.agents:
                        if hasattr(agent, "position") and new_loc == agent.position:
                            overlap_with_agent = True
                            break
                
                if not overlap_with_agent:
                    # 添加新灾难点
                    env.disaster_locations.add(new_loc)
                    break
            
            attempts += 1
        
        if attempts >= max_attempts and verbose:
            print("警告：无法找到有效的灾难点位置")
    
    if verbose:
        print(f"添加了{num_to_add}个灾难点，当前灾难点数量：{len(env.disaster_locations)}")
   


def main():
    """主函数"""
    args = parse_args()
    
    # 清屏，开始新的训练会话
    print("\033c", end="")  # 使用ANSI转义序列清屏
    
    # 覆盖配置文件中的模拟时间
    config.SIMULATION_TIME = args.steps
    
    # 打印简洁的训练配置信息
    print(f"\n{Colors.HEADER}{Colors.BOLD}======== MARL训练配置 ========{Colors.ENDC}")
    print(f"训练轮数: {Colors.YELLOW}{args.episodes}{Colors.ENDC}")
    print(f"每轮步数: {Colors.YELLOW}{args.steps}{Colors.ENDC}")
    print(f"模型保存: 每{Colors.YELLOW}{args.save_freq}{Colors.ENDC}轮")
    print(f"使用的预设灾难规模: {Colors.YELLOW}{config.DISASTER_PRESETS[config.DISASTER_SCALE]['name']}{Colors.ENDC}")
    print(f"网格大小: {Colors.YELLOW}{config.get_config_param('grid_size')}x{config.get_config_param('grid_size')}{Colors.ENDC}")
    print(f"救援人员: {Colors.YELLOW}{config.NUM_RESCUERS}{Colors.ENDC}名")
    
    # 打印灾难生成参数
    print(f"\n{Colors.BLUE}灾难生成参数:{Colors.ENDC}")
    print(f"  • 初始生成概率: {Colors.CYAN}{config.get_config_param('disaster_spawn_rate')}{Colors.ENDC}")
    print(f"  • 初始灾难数量: {Colors.CYAN}{config.get_config_param('initial_disasters')}{Colors.ENDC}")
    print(f"  • 最大灾难等级: {Colors.CYAN}{config.get_config_param('max_disaster_level')}{Colors.ENDC}")
    
    # 打印奖励设置
    print(f"\n{Colors.BLUE}奖励设置 (可在src/rl/marl_rescue.py中调整):{Colors.ENDC}")
    print(f"  • 完成救援 (COMPLETION_REWARD): {Colors.GREEN}+10.0{Colors.ENDC}")
    print(f"  • 高优先级任务 (PRIORITY_FACTOR): {Colors.GREEN}+任务等级*0.1{Colors.ENDC}")
    print(f"  • 协调避免重复 (COORDINATION_REWARD): {Colors.GREEN}+2.0{Colors.ENDC}")
    print(f"  • 救援进度 (PROGRESS_FACTOR): {Colors.GREEN}+进度*1.0{Colors.ENDC}")
    print(f"  • 时间惩罚 (TIME_PENALTY): {Colors.RED}-0.01/步{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}================================{Colors.ENDC}")
    
    # 初始化环境
    env = Environment(verbose=True, training_mode=True)  # 启用训练模式以激活数量上限控制
    
    # 创建MARL控制器
    marl = MARLController(
        env_or_grid_size=env.GRID_SIZE,
        num_rescuers=len(env.rescuers),
        hidden_dim=config.MARL_CONFIG["hidden_dim"],
        lr=config.MARL_CONFIG["learning_rate"],
        gamma=config.MARL_CONFIG["gamma"]
    )
    
    # 初始化环境缓存，用于救援人员协调
    marl._env_cache = [env]
    
    # 使用自定义训练循环替代train_marl函数
    rewards, success_rates, response_times = train_marl(
        env, 
        marl,
        args.episodes, 
        args.steps,
        with_verbose=False,
        save_freq=args.save_freq
    )
    
    # 计算和打印最终性能指标
    final_success_rate = success_rates[-1] if success_rates else 0
    final_response_time = response_times[-1] if response_times else 0
    final_reward = rewards[-1] if rewards else 0
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}======== MARL训练完成 ========{Colors.ENDC}")
    print(f"{Colors.BOLD}最终模型性能:{Colors.ENDC}")
    print(f"  • 平均奖励: {Colors.GREEN if final_reward > 0 else Colors.RED}{final_reward:.2f}{Colors.ENDC}")
    print(f"  • 救援成功率: {Colors.GREEN}{final_success_rate:.2f}{Colors.ENDC}")
    print(f"  • 平均响应时间: {Colors.CYAN}{final_response_time:.2f}{Colors.ENDC}")
    print(f"模型已保存到: {Colors.YELLOW}{config.MARL_CONFIG['model_save_path']}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}================================{Colors.ENDC}\n")
    
    # 使用高级可视化工具分析训练过程
    print("\n正在使用高级可视化工具分析训练过程...")
    
    # 读取元数据目录
    metadata_dir = os.path.join(SAVE_DIR, "metadata")
    if not os.path.exists(metadata_dir):
        print(f"元数据目录不存在: {metadata_dir}")
        return
    
    # 读取所有JSON元数据文件
    metadata_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('.json')])
    if not metadata_files:
        print(f"元数据目录中没有找到JSON文件: {metadata_dir}")
        return
    
    # 选择最新的元数据文件
    latest_metadata_file = os.path.join(metadata_dir, metadata_files[-1])
    print(f"使用最新元数据文件: {latest_metadata_file}")
    
    # 读取元数据文件
    with open(latest_metadata_file, 'r') as f:
        data = json.load(f)
    
    # 从元数据中获取快照信息
    metadata = data.get('metadata', [])
    if not metadata:
        print(f"元数据文件中没有找到有效数据: {latest_metadata_file}")
        return
    
    # 提取快照目录和文件名
    episode_str = os.path.basename(latest_metadata_file).replace("episode_", "").replace(".json", "")
    episode_num = int(episode_str)
    
    # 构建快照文件路径
    snapshot_dir = os.path.join(SAVE_DIR, "snapshots", f"episode_{episode_num:04d}")
    snapshot_file = os.path.join(snapshot_dir, "final_state.pkl")
    
    if not os.path.exists(snapshot_file):
        print(f"找不到快照文件: {snapshot_file}")
        return
    
    # 读取快照文件
    with open(snapshot_file, 'rb') as f:
        snapshot = pickle.load(f)
    
    # 准备可视化数据
    env_snapshots = [snapshot]
    progress_data = [(item['time_step'], item['success_rate']) for item in metadata]
    progress_data.sort(key=lambda x: x[0])
    
    # 创建可视化保存目录
    visualization_dir = os.path.join(SAVE_DIR, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 直接使用visualization.visualize进行可视化
    visualize(env_snapshots, progress_data)
    
    print(f"可视化结果已保存到: {visualization_dir}")




if __name__ == "__main__":
    main() 