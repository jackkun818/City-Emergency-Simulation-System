#!/usr/bin/env python3
"""
多智能体强化学习(MARL)训练脚本

用法:
  python train.py --episodes 1000 --steps 100 --save-freq 100

参数:
  --episodes   训练轮数 (默认: 1000)
  --steps      每轮训练步数 (默认: 100)
  --save-freq  保存频率，每多少轮保存一次模型 (默认: 100)
"""

import argparse
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

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# 导入相关模块
from core.environment import Environment
from core import config
from rl.marl_rescue import train_marl, MARLController

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
    parser.add_argument('--save-freq', type=int, default=100, help='保存频率，每多少轮保存一次模型 (默认: 100)')
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


def _remove_disasters_from_env(env, num_to_remove, verbose=False):
    """根据环境接口移除灾难点"""
    if hasattr(env, "disasters"):
        # 获取所有可移除的灾难点（不包括正在被救援的点）
        removable_disasters = []
        
        for pos, disaster in env.disasters.items():
            # 跳过已经被分配救援人员的灾难点
            is_targeted = False
            if hasattr(env, "rescuers"):
                for rescuer in env.rescuers:
                    if "target" in rescuer and rescuer["target"] == pos:
                        is_targeted = True
                        break
            
            # 只考虑未被分配且未冻结的灾难点
            if not is_targeted and not disaster.get("frozen_level", False) and not disaster.get("frozen_rescue", False):
                removable_disasters.append(pos)
        
        # 确保不会尝试移除超过实际可移除灾难点数量
        num_to_remove = min(num_to_remove, len(removable_disasters))
        
        # 随机选择要移除的灾难点
        if removable_disasters:
            disasters_to_remove = random.sample(removable_disasters, num_to_remove)
            
            # 移除选中的灾难点
            for disaster in disasters_to_remove:
                del env.disasters[disaster]
            
            if verbose:
                print(f"移除了{num_to_remove}个灾难点，当前灾难点数量：{len(env.disasters)}")


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


def custom_train_loop(env, controller, num_episodes, max_steps, with_verbose=False, save_freq=10):
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
    
    try:
        # 存储每个回合的奖励，用于计算平均奖励
        all_rewards = []
        success_rates = []
        response_times = []
        losses = []  # 添加损失追踪
        
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
                try:
                    if hasattr(env, 'update_disasters_silent'):
                        env.update_disasters_silent(current_time_step=env.current_time_step)
                    else:
                        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                            env.update_disasters(current_time_step=env.current_time_step)
                except Exception as e:
                    print(f"更新灾难状态时出错: {e}")
                
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
                    try:
                        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                            next_state, reward, done, info = env.step(rescuer_idx, action)
                    except Exception as e:
                        print(f"执行动作时出错: {e}")
                        next_state, reward, done = state, -1, False
                        info = {}
                    
                    # 处理奖励信息
                    total_reward += reward
                    if 'success' in info and info['success']:
                        success_count += 1
                    if 'response_time' in info and info['response_time'] > 0:  # 确保响应时间有效
                        total_response_time += info['response_time']
                        #print(f"救援成功！响应时间: {info['response_time']}，累计响应时间: {total_response_time}，成功次数: {success_count}")
                        
                    # 存储转换到经验回放缓冲区
                    controller.store_transition(state, action, reward, next_state, done, rescuer_idx)
                
                # 更新神经网络
                loss = controller.update_agents()
                episode_loss += loss  # 累计损失
                
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
                try:
                    if hasattr(controller, "get_reward_stats"):
                        reward_stats = controller.get_reward_stats(reset=False)
                        print(f"\n奖励分解:")
                        for reward_type, stats in reward_stats.items():
                            if stats["total"] != 0:  # 只显示非零奖励
                                print(f"• {reward_type}: 总计={stats['total']:.2f}, 平均={stats['avg']:.2f}")
                except Exception as e:
                    print(f"获取奖励分解时出错: {e}")
                
                print("==================================\n")
        
        return all_rewards, success_rates, response_times
    except Exception as e:
        import traceback
        print(f"训练过程中出错: {e}")
        traceback.print_exc()
        return [], [], []


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
    print(f"\n{Colors.BLUE}奖励设置:{Colors.ENDC}")
    print(f"  • 完成救援: {Colors.GREEN}+10.0{Colors.ENDC}")
    print(f"  • 高优先级任务: {Colors.GREEN}+任务等级/10{Colors.ENDC}")
    print(f"  • 协调避免重复: {Colors.GREEN}+2.0{Colors.ENDC}")
    print(f"  • 救援进度: {Colors.GREEN}+等级减少值{Colors.ENDC}")
    print(f"  • 时间惩罚: {Colors.RED}-0.1/步{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}================================{Colors.ENDC}")
    
    # 初始化环境
    env = Environment(verbose=True)  # 只在开始时显示详细配置
    
    # 创建MARL控制器
    marl = MARLController(
        grid_size=env.GRID_SIZE,
        num_rescuers=len(env.rescuers),
        hidden_dim=config.MARL_CONFIG["hidden_dim"],
        lr=config.MARL_CONFIG["learning_rate"],
        gamma=config.MARL_CONFIG["gamma"]
    )
    
    # 初始化环境缓存，用于救援人员协调
    marl._env_cache = [env]
    
    # 使用自定义训练循环替代train_marl函数
    rewards, success_rates, response_times = custom_train_loop(
        env, 
        marl,
        args.episodes, 
        args.steps,
        with_verbose=False,
        save_freq=args.save_freq
    )
    
    # 可视化训练结果
    visualize_training_results(rewards, success_rates, response_times)
    
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


if __name__ == "__main__":
    main() 