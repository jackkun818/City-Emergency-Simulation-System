import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib
from matplotlib.widgets import Slider, Button  # 导入Slider和Button控件
# 使用通用的英文字体，删除SimHei引用
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Tahoma', 'Verdana']
# 保留以下设置以确保减号符号正常显示
matplotlib.rcParams['axes.unicode_minus'] = False
import time  # 用于控制动画帧率

def visualize(env_snapshots, progress_data=None):
    """
    Enhanced Visualization:
    - 🚑 Display A* planned rescue paths
    - 🔴 Disaster points (severe) → 🟡 Disaster points (reduced) → 🟢 Rescue completed
    - 📊 Task progress bar
    - 📈 Rescue success rate curve
    """
    
    # 如果只有一个环境状态（向后兼容）
    if not isinstance(env_snapshots, list):
        env_snapshots = [env_snapshots]
        
    # 从环境快照中提取数据
    extracted_envs = []
    time_steps = []
    success_rates = []
    
    for snapshot in env_snapshots:
        if isinstance(snapshot, dict):
            # 新格式：包含env、time_step和success_rate
            extracted_envs.append(snapshot["env"])
            time_steps.append(snapshot["time_step"])
            success_rates.append(snapshot["success_rate"])
        else:
            # 旧格式：直接是环境对象
            extracted_envs.append(snapshot)
            
    # 如果没有提取出环境对象，使用旧的逻辑
    if not extracted_envs:
        extracted_envs = env_snapshots
        
    # 获取网格大小（从第一个快照）
    grid_size = extracted_envs[0].GRID_SIZE  # Get grid size
    
    # 创建图形并调整布局，为滑动条和成功率图表留出空间
    fig = plt.figure(figsize=(14, 12))
    
    # 创建网格、成功率图和滑动条的子图区域
    grid_ax = plt.subplot2grid((10, 2), (0, 0), rowspan=7, colspan=2)  # 主网格区域(70%)
    rate_ax = plt.subplot2grid((10, 2), (7, 0), rowspan=2, colspan=2)  # 成功率图区域(20%)
    slider_ax = plt.subplot2grid((10, 2), (9, 0), rowspan=1, colspan=2)  # 滑动条区域(10%)
    
    # 设置网格和固定边界
    grid_ax.set_xticks(range(grid_size))
    grid_ax.set_yticks(range(grid_size))
    grid_ax.set_xticklabels([])
    grid_ax.set_yticklabels([])
    grid_ax.grid(True)  # Enable grid lines
    
    # 设置固定的坐标轴范围，确保地图大小不变
    grid_ax.set_xlim(-0.5, grid_size - 0.5)
    grid_ax.set_ylim(grid_size - 0.5, -0.5)  # 反转y轴使坐标原点在左上角
    
    # 创建时间步滑动条
    time_slider = Slider(
        ax=slider_ax,
        label='Time Step',
        valmin=0,
        valmax=len(extracted_envs) - 1,
        valinit=0,
        valstep=1,
        color='skyblue'
    )
    
    # 初始显示第一帧
    current_frame = 0
    
    # 绘制成功率曲线
    def plot_success_rate():
        rate_ax.clear()
        
        # 如果有来自快照的成功率数据，优先使用
        if time_steps and success_rates:
            rate_ax.plot(time_steps, success_rates, marker="o", linestyle="-", 
                        color="b", label="Rescue Success Rate (30-step window)")
        # 否则使用传入的progress_data
        elif progress_data:
            times, rates = zip(*progress_data)
            rate_ax.plot(times, rates, marker="o", linestyle="-", 
                        color="b", label="Rescue Success Rate")
        
        rate_ax.set_xlabel("Time Step")
        rate_ax.set_ylabel("Success Rate")
        rate_ax.set_title("Rescue Success Rate Over Time")
        rate_ax.legend(loc='upper left')
        rate_ax.grid(True)
        
        # 设置y轴范围为0-1
        rate_ax.set_ylim(0, 1.05)
        
        # 如果当前时间步有值，在图上标记当前位置
        if current_frame < len(time_steps):
            current_time = time_steps[current_frame]
            # 在当前时间步画一条垂直线
            rate_ax.axvline(x=current_time, color='r', linestyle='--', alpha=0.7)
            
            # 如果有对应的成功率值，显示一个点
            if current_frame < len(success_rates):
                rate_ax.plot(current_time, success_rates[current_frame], 'ro', ms=10)
    
    # 绘制初始成功率图
    plot_success_rate()
    
    def update_plot(frame=None):
        """更新绘图函数，可以被滑动条或动画调用"""
        # 使用当前帧或传入的帧
        nonlocal current_frame
        if frame is not None:
            current_frame = int(frame)
            
        # 确保frame在快照范围内
        frame_idx = min(current_frame, len(extracted_envs) - 1)
        env = extracted_envs[frame_idx]  # 获取当前时间步的环境状态
        
        grid_ax.clear()  # Clear current plot
        grid_ax.set_xticks(range(grid_size))
        grid_ax.set_yticks(range(grid_size))
        grid_ax.set_xticklabels([])
        grid_ax.set_yticklabels([])
        grid_ax.grid(True)  # Re-enable grid lines
        grid_ax.set_title(f"City Emergency Rescue Simulation - Time Step {time_steps[frame_idx] if time_steps else frame_idx}")  # Set title
        
        # 重新设置固定的坐标轴范围
        grid_ax.set_xlim(-0.5, grid_size - 0.5)
        grid_ax.set_ylim(grid_size - 0.5, -0.5)

        # Record disaster point locations
        disaster_positions = {"high": [], "medium": [], "low": []}
        rescued_positions = []

        if env.disasters:  # Avoid error when disaster points are empty
            for (x, y), data in env.disasters.items():
                # 不显示已完成救援的灾情点
                if data.get("frozen_rescue", False):
                    continue
                
                # level=0且rescue_needed>0的点在显示完红叉后不再显示
                if data.get("frozen_level", False) and data.get("show_red_x", 0) == 0:
                    continue
                    
                if data["level"] > 5:
                    disaster_positions["high"].append((y, x))  # Severe disaster point
                elif data["level"] > 2:
                    disaster_positions["medium"].append((y, x))  # Medium disaster point
                elif data["level"] > 0:
                    disaster_positions["low"].append((y, x))  # Minor disaster point
                else:
                    # level=0但rescue_needed>0的情况，显示为特殊状态
                    rescued_positions.append((y, x))  # Level 0 but still needs rescue

        # Record rescuer positions & paths
        rescuer_positions = []
        rescuer_paths = []

        if env.rescuers:  # Avoid error when rescuer list is empty
            for rescuer in env.rescuers:
                rescuer_positions.append((rescuer["position"][1], rescuer["position"][0]))  # Record rescuer position
                if "path" in rescuer and rescuer["path"]:
                    rescuer_paths.extend([(p[1], p[0]) for p in rescuer["path"]])  # Record A* path

        # Draw disaster points (ensure at least one point exists, otherwise `plt.legend()` will warn)
        has_legend = False
        if disaster_positions["high"]:
            grid_ax.scatter(*zip(*disaster_positions["high"]), c="red", marker="o", s=100, label="High Severity Disaster")
            has_legend = True
        if disaster_positions["medium"]:
            grid_ax.scatter(*zip(*disaster_positions["medium"]), c="orange", marker="o", s=80, label="Medium Severity Disaster")
            has_legend = True
        if disaster_positions["low"]:
            grid_ax.scatter(*zip(*disaster_positions["low"]), c="yellow", marker="o", s=60, label="Low Severity Disaster")
            has_legend = True
        if rescued_positions:
            grid_ax.scatter(*zip(*rescued_positions), c="purple", marker="o", s=70, label="Level 0 (Needs More Rescue)")
            has_legend = True
            
        # 绘制红叉 - 标识救援失败的点(level=0但rescue_needed>0)
        red_x_positions = []
        for (x, y), data in env.disasters.items():
            if data.get("show_red_x", 0) > 0:
                red_x_positions.append((y, x))
                # 添加红叉标记文本，提高可见性
                grid_ax.text(y, x, 'X', color="red", fontsize=14, 
                              fontweight='bold', ha='center', va='center')
                
        # 确保有红叉位置才绘制，避免空列表错误
        if red_x_positions and len(red_x_positions) > 0:
            try:
                # 使用'x'标记并加大尺寸，增加线宽
                grid_ax.scatter(*zip(*red_x_positions), c="red", marker="x", s=250, 
                                linewidths=3, zorder=10, label="Failed Rescue Point")
                has_legend = True
            except Exception as e:
                print(f"Error drawing red X markers: {e}")

        # Draw rescuers
        if rescuer_positions:
            grid_ax.scatter(*zip(*rescuer_positions), c="blue", marker="s", s=100, label="Rescuers")
            has_legend = True

        # Draw A* planned paths
        if rescuer_paths:
            grid_ax.scatter(*zip(*rescuer_paths), c="cyan", marker=".", s=30, label="A* Path")
            has_legend = True

        # Label disaster level
        for (x, y), data in env.disasters.items():
            # 只为未冻结的灾情点显示level数字
            if not data.get("frozen_level", False) and not data.get("frozen_rescue", False):
                grid_ax.text(y, x, str(data["level"]), color="black", ha="center", va="center", fontsize=10)  # Display level on disaster points

        # Only add legend when there are markers, avoid `UserWarning`
        if has_legend:
            grid_ax.legend(loc='upper right')
            
        # 更新成功率图表，显示当前时间步
        plot_success_rate()
        
        fig.canvas.draw_idle()  # 重绘画布
        return grid_ax
    
    # 滑动条变化时的回调函数
    def slider_update(val):
        update_plot(val)
    
    # 连接滑动条事件
    time_slider.on_changed(slider_update)
    
    # 初始化显示第一帧
    update_plot(0)
    
    # 添加切换动画播放的按钮
    play_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    play_button = Button(play_ax, 'Play/Pause')
    
    # 动画控制 - 使用FuncAnimation而不是Timer
    anim = None
    animation_running = False
    
    def update_animation(frame):
        # 由FuncAnimation自动调用，不需要检查状态
        nonlocal current_frame
        try:
            current_frame = (current_frame + 1) % len(extracted_envs)
            time_slider.set_val(current_frame)
            # 滑动条的更新会触发update_plot，同时更新成功率图表
        except Exception as e:
            print(f"Animation error: {e}")
        return []  # 返回空列表，因为我们通过slider更新
    
    # 播放/暂停按钮回调
    def toggle_animation(event):
        nonlocal anim, animation_running
        
        # 防止重复点击造成的问题
        plt.pause(0.1)
        
        if animation_running:
            # 暂停动画
            try:
                if anim is not None:
                    anim.event_source.stop()
                    print("Animation paused")
            except Exception as e:
                print(f"Error pausing animation: {e}")
        else:
            # 开始动画
            try:
                if anim is None:
                    # 第一次创建动画
                    anim = animation.FuncAnimation(
                        fig, 
                        update_animation, 
                        frames=None,  # 无限帧
                        interval=200, 
                        blit=True,
                        repeat=True
                    )
                else:
                    # 重新启动动画
                    anim.event_source.start()
                
                print("Animation started")
            except Exception as e:
                print(f"Error starting animation: {e}")
        
        # 切换状态
        animation_running = not animation_running
    
    # 当窗口关闭时停止动画
    def on_close(event):
        nonlocal anim
        if anim is not None:
            try:
                anim.event_source.stop()
                print("Animation stopped due to window close")
            except:
                pass
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    # 使用环境快照的数量作为帧数
    frames_count = len(extracted_envs)
    play_button.on_clicked(toggle_animation)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    plt.show()  # Display figure
