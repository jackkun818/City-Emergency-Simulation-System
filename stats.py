import matplotlib.pyplot as plt
import matplotlib
import time
# Use generic English fonts, remove SimHei reference
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Tahoma', 'Verdana']
# Keep this setting to ensure minus signs display correctly
matplotlib.rcParams['axes.unicode_minus'] = False

def calculate_rescue_success_rate(disasters, window=30, current_time_step=None):
    """
    Calculate rescue success rate: number of successful rescue tasks / total tasks
    使用最近的N个已完成灾情点计算救援成功率，不考虑进行中的灾情点
    :param disasters: { (x, y): {"level": 0, "rescue_needed": 5, "time_step": 15, ...} }
    :param window: 要考虑的最近已完成灾情点数量
    :param current_time_step: 当前时间步（可选）
    :return: float Rescue success rate (between 0~1)
    """
    # 如果没有灾情点，返回0（表示还没有数据）
    if not disasters:
        return 0.0
    
    # 先筛选出所有已完成的灾情点（成功或失败）
    completed_disasters = [(pos, data) for pos, data in disasters.items() 
                          if data.get("frozen_rescue", False) or data.get("frozen_level", False)]
    
    # 如果没有已完成的灾情点，返回0
    if not completed_disasters:
        return 0.0
    
    # 找出所有有时间步记录的已完成灾情点
    timed_completed_disasters = [(pos, data) for pos, data in completed_disasters if "time_step" in data]
    
    # 如果有带时间步的已完成灾情点，按时间步从新到旧排序
    if timed_completed_disasters:
        sorted_completed_disasters = sorted(timed_completed_disasters, key=lambda x: x[1]["time_step"], reverse=True)
        
        # 取最近的N个已完成灾情点（N为窗口大小或可用已完成灾情点总数中较小的值）
        recent_count = min(window, len(sorted_completed_disasters))
        recent_completed_disasters = sorted_completed_disasters[:recent_count]
        
        # 计算这些灾情点中成功救援的数量
        successful = sum(1 for _, data in recent_completed_disasters if data.get("rescue_success", False))
        
        # 返回成功率
        return successful / len(recent_completed_disasters)
    
    # 如果没有时间记录的已完成灾情点，使用所有已完成的灾情点
    successful = sum(1 for _, data in completed_disasters if data.get("rescue_success", False))
    
    # 返回成功率
    return successful / len(completed_disasters)

def calculate_average_response_time(disasters):
    """
    Calculate average response time (time from task generation to rescue completion)
    :param disasters: { (x, y): {"start_time": 2, "end_time": 10} }
    :return: float Average response time
    """
    response_times = [d["end_time"] - d["start_time"] for d in disasters.values() if "end_time" in d]
    return sum(response_times) / len(response_times) if response_times else 0

def calculate_resource_utilization(rescuers, total_time):
    """
    Calculate resource utilization (rescue personnel task load)
    :param rescuers: List of rescue personnel, each containing {"id", "active_time": 10}
    :param total_time: Total simulation time
    :return: float Resource utilization (between 0~1)
    """
    total_rescue_time = sum(rescuer["active_time"] for rescuer in rescuers)
    max_possible_time = total_time * len(rescuers)
    return total_rescue_time / max_possible_time if max_possible_time > 0 else 0

def verify_rescue_stats(disasters, progress_data, time_step):
    """
    验证救援统计数据是否正确
    :param disasters: 当前灾情字典
    :param progress_data: 已记录的救援进度数据
    :param time_step: 当前时间步
    :return: None
    """
    # 检验是否有足够的数据进行验证
    if not progress_data:
        print("尚无进度数据，无法验证")
        return
    
    # 获取记录的最后一个成功率
    last_recorded_time, last_recorded_rate = progress_data[-1]
    
    # 尝试重新计算最近N个已完成灾情点的成功率并比较
    window_size = 30  # 默认窗口大小
    
    print(f"验证时间步 {time_step} 的统计数据:")
    
    # 先筛选出所有已完成的灾情点（成功或失败）
    completed_disasters = [(pos, data) for pos, data in disasters.items() 
                          if data.get("frozen_rescue", False) or data.get("frozen_level", False)]
    
    # 如果没有已完成的灾情点，预期成功率为0
    if not completed_disasters:
        expected_rate = 0.0
        print(f"  - 记录的成功率: {last_recorded_rate:.4f}")
        print(f"  - 预期的成功率: {expected_rate:.4f}")
        
        if abs(expected_rate - last_recorded_rate) > 0.001:  # 允许小误差
            print(f"  ⚠️ 警告: 成功率计算可能有误！(差异: {abs(expected_rate - last_recorded_rate):.4f})")
        else:
            print(f"  ✅ 成功率计算正确")
        return
    
    # 找出所有有时间步记录的已完成灾情点
    timed_completed_disasters = [(pos, data) for pos, data in completed_disasters if "time_step" in data]
    
    # 如果有带时间步的已完成灾情点，按时间步从新到旧排序
    if timed_completed_disasters:
        sorted_completed_disasters = sorted(timed_completed_disasters, key=lambda x: x[1]["time_step"], reverse=True)
        
        # 取最近的N个已完成灾情点
        recent_count = min(window_size, len(sorted_completed_disasters))
        recent_completed_disasters = sorted_completed_disasters[:recent_count]
        
        # 计算这些灾情点中成功救援的数量
        successful = sum(1 for _, data in recent_completed_disasters if data.get("rescue_success", False))
        
        # 计算预期成功率
        expected_rate = successful / len(recent_completed_disasters)
    else:
        # 如果没有时间记录的已完成灾情点，使用所有已完成的灾情点
        successful = sum(1 for _, data in completed_disasters if data.get("rescue_success", False))
        
        # 计算预期成功率
        expected_rate = successful / len(completed_disasters)
    
    print(f"  - 记录的成功率: {last_recorded_rate:.4f}")
    print(f"  - 预期的成功率: {expected_rate:.4f}")
    
    # 检查误差
    if abs(expected_rate - last_recorded_rate) > 0.001:  # 允许小误差
        print(f"  ⚠️ 警告: 成功率计算可能有误！(差异: {abs(expected_rate - last_recorded_rate):.4f})")
    else:
        print(f"  ✅ 成功率计算正确")

def show_disaster_distribution(disasters, window=30, current_time_step=None):
    """
    显示灾情点的分布，用于调试和监控最近N个已完成灾情点的统计效果
    :param disasters: 灾情点字典
    :param window: 要考虑的最近已完成灾情点数量
    :param current_time_step: 当前时间步
    :return: None
    """
    if not disasters:
        print("没有灾情点数据可显示")
        return
    
    print(f"\n📊 灾情点时间分布 (考虑最近 {window} 个已完成灾情点):")
    
    # 统计所有灾情点
    print("【所有灾情点统计】")
    total = len(disasters)
    success = sum(1 for d in disasters.values() if d.get("rescue_success", False))
    fail = sum(1 for d in disasters.values() if d.get("frozen_level", False) and not d.get("rescue_success", False))
    active = total - success - fail
    
    print(f"  总数: {total}, 成功: {success}, 失败: {fail}, 进行中: {active}")
    
    # 筛选出所有已完成的灾情点（成功或失败）
    completed_disasters = [(pos, data) for pos, data in disasters.items() 
                          if data.get("frozen_rescue", False) or data.get("frozen_level", False)]
    
    if not completed_disasters:
        print("\n没有已完成的灾情点，成功率为0")
        return
    
    # 找出所有有时间步记录的已完成灾情点
    timed_completed_disasters = [(pos, data) for pos, data in completed_disasters if "time_step" in data]
    
    # 如果有带时间步的已完成灾情点
    if timed_completed_disasters:
        # 按时间排序
        sorted_completed_disasters = sorted(timed_completed_disasters, key=lambda x: x[1]["time_step"], reverse=True)
        
        # 计算要显示的灾情点数量
        display_count = min(window, len(sorted_completed_disasters))
        recent_completed_disasters = sorted_completed_disasters[:display_count]
        
        print("\n【最近已完成灾情点详情】")
        print("  序号 |  时间步  | 状态")
        print("  " + "-" * 25)
        
        # 计算最近N个已完成灾情点的统计
        recent_total = len(recent_completed_disasters)
        recent_success = sum(1 for _, d in recent_completed_disasters if d.get("rescue_success", False))
        recent_fail = recent_total - recent_success
        
        # 显示每个已完成灾情点的详情
        for i, (pos, data) in enumerate(recent_completed_disasters):
            time_step = data.get("time_step", "未知")
            
            if data.get("rescue_success", False):
                status = "✅ 成功"
            else:
                status = "❌ 失败"
                
            print(f"  {i+1:4d} | {time_step:8d} | {status}")
        
        print("  " + "-" * 25)
        print(f"  最近{display_count}个已完成灾情点统计: 总数: {recent_total}, 成功: {recent_success}, 失败: {recent_fail}")
        
        # 计算已完成灾情点的成功率
        success_rate = recent_success / recent_total
        print(f"  已完成灾情点成功率: {success_rate:.4f}")
    else:
        # 如果没有时间记录的已完成灾情点，使用所有已完成的灾情点
        total_completed = len(completed_disasters)
        total_success = sum(1 for _, d in completed_disasters if d.get("rescue_success", False))
        
        print("\n【所有已完成灾情点统计】")
        print(f"  总数: {total_completed}, 成功: {total_success}, 失败: {total_completed - total_success}")
        
        # 计算成功率
        success_rate = total_success / total_completed
        print(f"  已完成灾情点成功率: {success_rate:.4f}")
        print("  注意: 这些灾情点没有时间步信息")

def plot_rescue_progress(progress_data):
    """
    Plot rescue progress curve
    :param progress_data: [(time, success_rate)]
    """
    times, rates = zip(*progress_data)
    plt.plot(times, rates, marker="o", linestyle="-", color="b", label="Rescue Success Rate")
    plt.xlabel("Time")
    plt.ylabel("Success Rate")
    plt.title("Rescue Success Rate Over Time")
    plt.legend()
    plt.grid()
    plt.show()
