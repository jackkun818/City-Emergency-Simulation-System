import matplotlib.pyplot as plt
import matplotlib
# Use generic English fonts, remove SimHei reference
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Tahoma', 'Verdana']
# Keep this setting to ensure minus signs display correctly
matplotlib.rcParams['axes.unicode_minus'] = False

def calculate_rescue_success_rate(disasters):
    """
    Calculate rescue success rate: number of successful rescue tasks / total tasks
    :param disasters: { (x, y): {"level": 0, "rescue_needed": 5, "start_time": 2, "end_time": 10} }
    :return: float Rescue success rate (between 0~1)
    """
    # 计算总灾情点数，包括已冻结的
    total_disasters = len(disasters)
    
    # 计算成功救援的灾情点数量 (有rescue_success=True标记的)
    successful_rescues = sum(1 for d in disasters.values() if d.get("rescue_success", False))
    
    # 如果没有灾情点，返回100%成功率
    return successful_rescues / total_disasters if total_disasters > 0 else 1.0

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
    # 统计当前存在的灾情数量（包括已冻结的）
    total_disasters = len(disasters)
    
    # 统计成功救援的灾情点
    successful = sum(1 for d in disasters.values() if d.get("rescue_success", False))
    
    # 统计自然结束但未完成救援的灾情点
    failed = sum(1 for d in disasters.values() 
                if d.get("frozen_level", False) and not d.get("rescue_success", False))
    
    # 计算当前救援成功率
    current_rate = successful / total_disasters if total_disasters > 0 else 1.0
    
    # 比较与记录的最后一个救援成功率
    if progress_data:
        last_recorded_time, last_recorded_rate = progress_data[-1]
        print(f"时间步 {time_step}:")
        print(f"  - 总灾情点: {total_disasters}")
        print(f"  - 成功救援的灾情点: {successful}")
        print(f"  - 未成功救援的灾情点: {failed}")
        print(f"  - 活跃灾情点: {total_disasters - successful - failed}")
        print(f"  - 计算的成功率: {current_rate:.4f}")
        print(f"  - 记录的成功率: {last_recorded_rate:.4f}")
        
        if abs(current_rate - last_recorded_rate) > 0.001:  # 允许小误差
            print(f"  ⚠️ 警告: 成功率计算可能有误！")
        else:
            print(f"  ✅ 成功率计算正确")
    
    # 建议在main.py中调用此函数，位置在记录progress_data后、清理完成灾情前

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
