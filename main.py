import time
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入核心模块
from core.environment import Environment
from core import config
from core.rescue_execution import execute_rescue

# 导入RL模块
from rl.marl_integration import dispatch_rescue_tasks, get_algorithm_name

# 导入工具和可视化模块
from utils.stats import (
    calculate_rescue_success_rate,
    calculate_average_response_time,
    calculate_resource_utilization,
    verify_rescue_stats,
    show_disaster_distribution
)
from visualization.visualization import visualize

import copy


def select_disaster_scale():
    """
    选择灾难规模并显示当前配置
    
    现在直接使用config.py中的配置，不再需要用户输入
    """
    # 显示当前灾难规模的信息
    if config.DISASTER_SCALE in config.DISASTER_PRESETS:
        preset = config.DISASTER_PRESETS[config.DISASTER_SCALE]
        print(f"使用{preset['name']} - 网格大小({preset['grid_size']}x{preset['grid_size']}), "
              f"灾情生成概率({preset['disaster_spawn_rate']}), 衰减步数{preset['spawn_rate_decay_steps']}")  
    else:
        print(f"警告：未知的灾难规模设置：{config.DISASTER_SCALE}")
    
    print(f"救援人员数量: {config.NUM_RESCUERS}")
    print(f"任务分配算法: {config.TASK_ALLOCATION_ALGORITHM} ({get_algorithm_name()})")
    print("注意：要修改这些设置，请直接编辑config.py文件")
    
    # 以下是原始的用户输入代码，现在被注释掉
    """
    while True:
        try:
            choice = int(input("请选择灾难规模 (0-3): "))
            if 0 <= choice <= 3:
                config.DISASTER_SCALE = choice
                break
            else:
                print("无效选择，请输入0-3之间的数字")
        except ValueError:
            print("无效输入，请输入0-3之间的数字")
    
    # 单独设置救援人员数量
    while True:
        try:
            rescuer_count = int(input(f"请输入救援人员数量 (默认为{config.NUM_RESCUERS}): ") or config.NUM_RESCUERS)
            if rescuer_count > 0:
                config.NUM_RESCUERS = rescuer_count
                break
            else:
                print("救援人员数量必须大于0")
        except ValueError:
            print("无效输入，请输入一个正整数")
    """


def main():
    """
    城市应急救援模拟主函数：
    1. 初始化环境（城市网格、灾情点、救援人员）
    2. 运行灾情模拟 & 救援调度
    3. 可视化救援过程
    4. 计算救援统计数据
    """
    
    # 选择灾难规模
    select_disaster_scale()

    print("🚀 Initializing urban rescue simulation environment...")  # 输出初始化城市救援模拟环境

    try:
        # 使用新的配置系统初始化环境，不再直接传递参数
        env = Environment()
    except TypeError as e:
        print(f"❌ Environment initialization failed: {e}")  # 输出环境初始化失败的错误信息
        print("⚠️ Please check if `environment.py` supports the new configuration system.")  # 提示用户检查 `environment.py` 文件
        return  # 终止程序

    # 初始化数据收集
    progress_data = []  # 初始化进度数据
    env_snapshots = []  # 保存每个时间步的环境快照

    # 开始救援模拟循环
    for time_step in range(config.SIMULATION_TIME):
        print(f"\n🕒 Time step: {time_step}")  # 输出当前时间步

        # 1️⃣ 更新灾情信息
        try:
            env.update_disasters(current_time_step=time_step)  # 更新灾情状态
        except AttributeError:
            print("❌ Error: `update_disasters()` method does not exist. Please check `environment.py`.")  # 输出错误提示
            return  # 终止程序

        # 2️⃣ 任务分配（智能调度救援任务）
        # 使用环境中的网格大小而不是直接用config中的
        # 使用集成的任务分配方法，替代原来的hybrid_rescue_dispatch
        dispatch_rescue_tasks(env.rescuers, env.disasters, env.GRID_SIZE, current_time_step=time_step)

        # 3️⃣ 执行救援任务（人员前往灾情点 & 进行救援）
        execute_rescue(env.rescuers, env.disasters, env.GRID_SIZE, current_time_step=time_step)  # 让救援人员前往目标点执行救援

        # 4️⃣ 记录救援进度（用于绘制成功率曲线）
        success_rate = calculate_rescue_success_rate(env.disasters, window=config.STATS_WINDOW_SIZE, current_time_step=time_step)  # 使用配置文件中的窗口大小
        progress_data.append((time_step, success_rate))  # 记录时间步和救援成功率
        
        # 验证救援统计数据（每10个时间步验证一次，避免输出过多）
        if time_step % 10 == 0:
            verify_rescue_stats(env.disasters, progress_data, time_step)
            # 显示灾情点时间分布（用于调试）
            show_disaster_distribution(env.disasters, window=config.STATS_WINDOW_SIZE, current_time_step=time_step)
        
        # 不再删除灾情点，而是在update_disasters中确保不更新已冻结的灾情点
        
        # 保存当前环境状态的快照和成功率数据
        env_snapshots.append({
            "env": copy.deepcopy(env),
            "time_step": time_step,
            "success_rate": success_rate
        })

        # 5️⃣ 等待短暂时间，模拟现实救援节奏（可选）
        time.sleep(0.1)  # 休眠 0.1 秒，模拟真实救援节奏

    print("\n✅ Rescue tasks completed. Analyzing statistics...")  # 输出救援任务完成，并开始分析统计数据

    # 计算最终统计数据
    final_success_rate = calculate_rescue_success_rate(env.disasters)  # 计算最终救援成功率
    avg_response_time = calculate_average_response_time(env.disasters)  # 计算平均响应时间
    resource_utilization = calculate_resource_utilization(env.rescuers, config.SIMULATION_TIME)  # 计算资源利用率

    # 输出最终统计结果
    print(f"\n📊 Final statistics:")  # 输出统计结果标题
    print(f"   - Rescue success rate: {final_success_rate * 100:.2f}%")  # 输出救援成功率
    print(f"   - Average response time: {avg_response_time:.2f} time units")  # 输出平均响应时间
    print(f"   - Resource utilization: {resource_utilization * 100:.2f}%")  # 输出资源利用率

    # 6️⃣ 可视化救援过程（包含救援成功率曲线）
    visualize(env_snapshots, progress_data)  # 传递环境快照列表和进度数据


# 运行主程序
if __name__ == "__main__":
    main()
