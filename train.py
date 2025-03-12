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
    parser.add_argument('--episodes', type=int, default=1000, help='训练轮数 (默认: 1000)')
    parser.add_argument('--steps', type=int, default=100, help='每轮训练步数 (默认: 100)')
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


def adjust_disaster_settings(env, step, max_steps, verbose=False):
    """
    根据训练进度动态调整灾难设置
    
    训练分为三个阶段：
    1. 前1/3: 灾难初期，高频率灾难点(0.5)，不少于20个灾难点
    2. 中间1/3: 灾难中期，中等频率灾难点(0.3)，5-20个灾难点
    3. 最后1/3: 灾难后期，低频率灾难点(0.1)，不超过5个灾难点
    """
    # 计算当前训练阶段 (0-2)
    stage = int(3 * step / max_steps)
    if stage >= 3:  # 防止可能的越界
        stage = 2
    
    # 直接在配置中设置灾难生成概率，而不仅仅是在预设中
    original_spawn_rate = config.get_config_param("disaster_spawn_rate")
    
    # 确保关闭灾难概率衰减
    config.DISASTER_PRESETS[config.DISASTER_SCALE]["enable_spawn_rate_decay"] = False
    
    # 根据阶段设置灾难生成概率
    if stage == 0:  # 前1/3: 灾难初期
        target_rate = 0.5
        min_disasters = 20
        # 如果灾难点数量不足，则提高生成概率
        if len(env.disasters) < min_disasters:
            new_rate = max(target_rate, original_spawn_rate)
        else:
            new_rate = target_rate
            
    elif stage == 1:  # 中间1/3: 灾难中期
        target_rate = 0.3
        min_disasters = 5
        max_disasters = 20
        # 根据当前灾难点数量调整生成概率
        if len(env.disasters) < min_disasters:
            new_rate = 0.4  # 略高一些
        elif len(env.disasters) > max_disasters:
            new_rate = 0.2  # 略低一些
        else:
            new_rate = target_rate
            
    else:  # 最后1/3: 灾难后期
        target_rate = 0.1
        max_disasters = 5
        # 如果灾难点数量超过限制，则降低生成概率
        if len(env.disasters) > max_disasters:
            new_rate = 0.05  # 非常低
        else:
            new_rate = target_rate
    
    # 直接设置灾难生成概率 - 同时设置预设和全局参数
    config.DISASTER_PRESETS[config.DISASTER_SCALE]["disaster_spawn_rate"] = new_rate
    
    # 确保config.py中的函数能够接收到新的概率值
    # 这需要设置适当的全局变量
    if hasattr(config, "_CUSTOM_DISASTER_SPAWN_RATE"):
        config._CUSTOM_DISASTER_SPAWN_RATE = new_rate
    
    # 打印当前阶段和设置（每10步打印一次）
    if verbose and step % 10 == 0:
        stage_names = ["初期", "中期", "后期"]
        print(f"训练阶段: 灾难{stage_names[stage]} | "
              f"灾难点数量: {len(env.disasters)} | "
              f"生成概率: {new_rate:.2f}")


def custom_train_loop(env, controller, num_episodes, max_steps, with_verbose=False, save_freq=10):
    """
    自定义训练循环，用于MARL训练
    """
    print("开始MARL训练过程...")
    print("-------------------------------------------")
    print("训练分为三个阶段：")
    print(" - 阶段1：灾难初期，生成概率约0.5，至少20个灾难点")
    print(" - 阶段2：灾难中期，生成概率约0.3，保持5-20个灾难点")
    print(" - 阶段3：灾难后期，生成概率约0.1，不超过5个灾难点")
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
                    
                # 调整灾难设置
                adjust_disaster_settings(env, step, max_steps, verbose=with_verbose)
                
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
                
                current_disasters = len(env.disasters)
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