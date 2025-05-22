#!/usr/bin/env python3
"""
Training Data Visualization Browser - Standalone Window Application

For browsing and visualizing saved training metadata
"""

import os
import json
import pickle
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use('TkAgg')  # Set Matplotlib backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import numpy as np
import sys
import re

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import visualization module
from src.visualization import visualize

# Set save directory
SAVE_DIR = os.path.join(project_root, 'train_visualization_save')

# Default data directory
DEFAULT_DATA_DIR = os.path.join(SAVE_DIR, 'metadata')

class TrainingDataBrowser:
    def __init__(self, root, data_dir=DEFAULT_DATA_DIR):
        self.root = root
        self.data_dir = data_dir
        self.selected_file = None
        self.current_fig = None
        self.current_canvas = None
        self.update_timer_id = None
        
        # Set window title
        self.root.title("Training Data Visualization Browser")
        
        # Setup UI
        self.setup_ui()
        
        # Load metadata files
        self.load_metadata_files()
    
    def setup_ui(self):
        """Setup user interface"""
        # Create split window
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - File browser
        self.browser_frame = ttk.LabelFrame(self.main_paned, text="Training Data Files")
        self.main_paned.add(self.browser_frame, weight=1)
        
        # Right panel - Visualization area
        self.viz_frame = ttk.LabelFrame(self.main_paned, text="Visualization")
        self.main_paned.add(self.viz_frame, weight=4)
        
        # Set up file browser content
        self.setup_browser()
        
        # Set up visualization area content
        self.setup_viz_area()
        
        # Set up bottom control bar
        self.setup_control_bar()
    
    def setup_browser(self):
        """Set up file browser area"""
        # Top operation buttons
        self.browser_buttons = ttk.Frame(self.browser_frame)
        self.browser_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        # Refresh button
        self.refresh_btn = ttk.Button(self.browser_buttons, text="Refresh", command=self.load_metadata_files)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Choose directory button
        self.dir_btn = ttk.Button(self.browser_buttons, text="Select Directory", command=self.choose_data_dir)
        self.dir_btn.pack(side=tk.RIGHT, padx=5)
        
        # File tree view
        self.file_tree_frame = ttk.Frame(self.browser_frame)
        self.file_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tree view
        self.file_tree = ttk.Treeview(self.file_tree_frame, selectmode="browse")
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(self.file_tree_frame, orient="vertical", command=self.file_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Set column headers
        self.file_tree["columns"] = ("episode", "steps", "success_rate")
        self.file_tree.column("#0", width=50, minwidth=50, stretch=tk.NO)
        self.file_tree.column("episode", width=80, minwidth=80, stretch=tk.NO)
        self.file_tree.column("steps", width=80, minwidth=80, stretch=tk.NO)
        self.file_tree.column("success_rate", width=100, minwidth=100, stretch=tk.NO)
        
        self.file_tree.heading("#0", text="ID", anchor=tk.W)
        self.file_tree.heading("episode", text="Episode", anchor=tk.W)
        self.file_tree.heading("steps", text="Steps", anchor=tk.W)
        self.file_tree.heading("success_rate", text="Success Rate", anchor=tk.W)
        
        # Bind selection event
        self.file_tree.bind("<<TreeviewSelect>>", self.on_file_select)
    
    def setup_viz_area(self):
        """Set up visualization area"""
        # Create tabs
        self.viz_notebook = ttk.Notebook(self.viz_frame)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create basic info tab
        self.info_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.info_tab, text="Basic Info")
        
        # Create metrics chart tab
        self.metrics_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.metrics_tab, text="Training Metrics")
        
        # Create environment visualization tab
        self.env_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.env_tab, text="Environment Visualization")
        
        # Setup basic info tab content
        self.info_text = tk.Text(self.info_tab, wrap=tk.WORD, padx=10, pady=10)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Setup metrics chart tab content
        self.metrics_frame = ttk.Frame(self.metrics_tab)
        self.metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup environment visualization tab content
        self.env_frame = ttk.Frame(self.env_tab)
        self.env_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bind tab selection event
        self.viz_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def setup_control_bar(self):
        """Set up bottom control bar"""
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.control_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Visualization button
        self.visualize_btn = ttk.Button(self.control_frame, text="Visualize Selected Data", command=self.visualize_selected, state=tk.DISABLED)
        self.visualize_btn.pack(side=tk.RIGHT, padx=5)
        
        # Export button
        self.export_btn = ttk.Button(self.control_frame, text="Export Figures", command=self.export_figures, state=tk.DISABLED)
        self.export_btn.pack(side=tk.RIGHT, padx=5)
    
    def load_metadata_files(self):
        """Load metadata file list"""
        # Clear existing tree
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            # If directory doesn't exist, try to create it
            parent_dir = os.path.dirname(self.data_dir)
            if os.path.exists(parent_dir):
                os.makedirs(self.data_dir, exist_ok=True)
                self.status_label.config(text=f"Created data directory: {self.data_dir}")
            else:
                # If parent directory doesn't exist, prompt to select valid directory
                self.status_label.config(text="Data directory doesn't exist. Please select a valid directory.")
                self.root.after(100, self.choose_data_dir)
                return
        
        # Get all JSON metadata files
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        files.sort()  # Sort by filename
        
        if not files:
            self.status_label.config(text=f"No metadata files found in directory {self.data_dir}")
            return
        
        # Add files to tree
        for i, file in enumerate(files):
            # Extract episode info from filename
            episode_str = file.replace("episode_", "").replace(".json", "")
            try:
                episode_num = int(episode_str)
            except ValueError:
                # If cannot convert to integer, use default value
                episode_num = i + 1
            
            # Read file to get basic info
            with open(os.path.join(self.data_dir, file), 'r') as f:
                data = json.load(f)
            
            # Get success rate and total steps of the last step
            metadata = data.get('metadata', [])
            max_steps = len(metadata)
            success_rate = data.get('metrics', {}).get('success_rates', [0])[-1]
            
            # Add to tree
            self.file_tree.insert("", tk.END, text=str(i+1), 
                                 values=(f"Episode {episode_num}", 
                                         f"{max_steps} steps",
                                         f"{success_rate:.2f}"),
                                 tags=(file,))
        
        # Update status
        self.status_label.config(text=f"Loaded {len(files)} training data files")
    
    def choose_data_dir(self):
        """Choose data directory"""
        dir_path = filedialog.askdirectory(initialdir=self.data_dir, title="Select Metadata Directory")
        if dir_path:
            self.data_dir = dir_path
            self.load_metadata_files()
    
    def on_file_select(self, event):
        """Handle file selection"""
        selection = self.file_tree.selection()
        if selection:
            # Get selected file
            item = selection[0]
            file = self.file_tree.item(item)['tags'][0]
            self.selected_file = os.path.join(self.data_dir, file)
            
            # Enable visualization button
            self.visualize_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            
            # Show basic info
            self.show_basic_info(self.selected_file)
            
            # Plot metrics
            with open(self.selected_file, 'r') as f:
                data = json.load(f)
            self.plot_metrics(data.get('metrics', {}))
    
    def on_tab_changed(self, event):
        """Handle tab change"""
        # Get current tab
        current_tab = self.viz_notebook.select()
        tab_name = self.viz_notebook.tab(current_tab, "text")
        
        if tab_name == "Environment Visualization":
            # Only trigger visualization if a file is actually selected
            if getattr(self, 'selected_file', None):
                self.visualize_selected()
    
    def show_basic_info(self, file_path):
        """Show basic information of selected file"""
        # Clear text
        self.info_text.delete(1.0, tk.END)
        
        try:
            # Read file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract basic information
            env_config = data.get('env_config', {})
            metadata = data.get('metadata', [])
            metrics = data.get('metrics', {})
            
            # Display information
            self.info_text.insert(tk.END, "===== Training Episode Information =====\n\n")
            
            # Environment configuration
            self.info_text.insert(tk.END, "Environment Configuration:\n")
            self.info_text.insert(tk.END, f"• Grid Size: {env_config.get('grid_size', 'N/A')}\n")
            self.info_text.insert(tk.END, f"• Number of Rescuers: {env_config.get('num_rescuers', 'N/A')}\n")
            self.info_text.insert(tk.END, f"• Maximum Steps: {env_config.get('max_steps', 'N/A')}\n\n")
            
            # Episode statistics
            if metadata:
                last_meta = metadata[-1]
                self.info_text.insert(tk.END, "Episode Statistics:\n")
                self.info_text.insert(tk.END, f"• Total Steps: {len(metadata)}\n")
                self.info_text.insert(tk.END, f"• Disaster Count: {last_meta.get('disaster_count', 'N/A')}\n")
                self.info_text.insert(tk.END, f"• Success Count: {last_meta.get('success_count', 'N/A')}\n")
                self.info_text.insert(tk.END, f"• Total Reward: {last_meta.get('total_reward', 'N/A')}\n")
                self.info_text.insert(tk.END, f"• Average Reward: {last_meta.get('avg_reward', 'N/A')}\n")
                self.info_text.insert(tk.END, f"• Average Response Time: {last_meta.get('avg_response_time', 'N/A')}\n\n")
            
            # Final metrics
            self.info_text.insert(tk.END, "Final Metrics:\n")
            if metrics:
                self.info_text.insert(tk.END, f"• Final Success Rate: {metrics.get('success_rates', [0])[-1]:.2f}\n")
                self.info_text.insert(tk.END, f"• Final Average Reward: {metrics.get('rewards', [0])[-1]:.2f}\n")
                self.info_text.insert(tk.END, f"• Final Average Response Time: {metrics.get('response_times', [0])[-1]:.2f}\n")
            
        except Exception as e:
            self.info_text.insert(tk.END, f"Error loading file: {str(e)}")
    
    def plot_metrics(self, metrics):
        """Plot training metrics"""
        # Clear previous plots
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # Check if we have any metrics data
        has_data = False
        for key in ['rewards', 'success_rates', 'response_times', 'losses']:
            if key in metrics and metrics[key] and len(metrics[key]) > 0:
                has_data = True
                break
        
        if not has_data:
            # Display message if no data available
            no_data_label = ttk.Label(self.metrics_frame, 
                                     text="No metrics data available for this episode",
                                     font=('Arial', 12))
            no_data_label.pack(expand=True, pady=50)
            return
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 8))
        
        # Configure subplots
        subplot_count = 0
        rows = 2
        cols = 2
        
        # Plot rewards
        if 'rewards' in metrics and metrics['rewards'] and len(metrics['rewards']) > 0:
            subplot_count += 1
            ax1 = fig.add_subplot(rows, cols, subplot_count)
            ax1.plot(metrics['rewards'], '-b', label='Reward')
            ax1.set_title('Average Reward')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
            
            # Add mean line
            mean_reward = sum(metrics['rewards']) / len(metrics['rewards'])
            ax1.axhline(y=mean_reward, color='r', linestyle='--', alpha=0.7)
            ax1.text(0, mean_reward, f' Mean: {mean_reward:.2f}', color='r')
            
        # Plot success rates
        if 'success_rates' in metrics and metrics['success_rates'] and len(metrics['success_rates']) > 0:
            subplot_count += 1
            ax2 = fig.add_subplot(rows, cols, subplot_count)
            ax2.plot(metrics['success_rates'], '-g', label='Success Rate')
            ax2.set_title('Rescue Success Rate')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Success Rate')
            ax2.grid(True)
            
            # Set y-axis range for success rate
            ax2.set_ylim(0, 1.05)
            
        # Plot response times
        if 'response_times' in metrics and metrics['response_times'] and len(metrics['response_times']) > 0:
            subplot_count += 1
            ax3 = fig.add_subplot(rows, cols, subplot_count)
            ax3.plot(metrics['response_times'], '-r', label='Response Time')
            ax3.set_title('Average Response Time')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Seconds')
            ax3.grid(True)
            
        # Plot losses
        if 'losses' in metrics and metrics['losses'] and len(metrics['losses']) > 0:
            subplot_count += 1
            ax4 = fig.add_subplot(rows, cols, subplot_count)
            ax4.plot(metrics['losses'], '-m', label='Loss')
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss')
            ax4.grid(True)
            
        # Adjust layout
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, self.metrics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        toolbar_frame = ttk.Frame(self.metrics_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
    
    def visualize_selected(self):
        """为选中的训练数据提供完全交互式可视化"""
        if not hasattr(self, 'selected_file'):
            raise ValueError("请先选择一个训练文件")

        try:
            # 清除环境可视化标签页中的所有部件，确保先停止所有定时器
            if hasattr(self, 'update_timer_id') and self.update_timer_id:
                self.root.after_cancel(self.update_timer_id)
                self.update_timer_id = None
            
            # 关闭所有已存在的matplotlib图表，防止资源冲突
            plt.close('all')
            
            # 清除所有现有组件
            for widget in self.env_frame.winfo_children():
                widget.destroy()
            
            # 确保在嵌入模式下使用正确的Matplotlib后端
            matplotlib.use("TkAgg", force=False)
            
            # 在开始新的可视化前，请求垃圾回收
            import gc
            gc.collect()
            
            # 加载选中文件的元数据
            with open(self.selected_file, 'r') as f:
                data = json.load(f)
            
            # 提取环境配置和元数据信息
            env_config = data.get('env_config', {})
            metadata_entries = data.get('metadata', [])
            
            # 确保我们有可视化数据
            if not metadata_entries:
                raise ValueError(f"在文件 {os.path.basename(self.selected_file)} 中没有找到可视化数据")
            
            # 提取剧集编号以找到对应的快照目录
            filename = os.path.basename(self.selected_file)
            match = re.search(r'episode_(\d+)', filename)
            if not match:
                raise ValueError(f"无法从文件名 {filename} 中提取剧集编号")
            
            episode_num = int(match.group(1))
            snapshot_dir = os.path.join(SAVE_DIR, "snapshots", f"episode_{episode_num:04d}")
            
            # 确认快照目录存在
            if not os.path.exists(snapshot_dir):
                raise FileNotFoundError(f"找不到快照目录: {snapshot_dir}")
            
            # 确认最终快照文件存在
            final_snapshot_file = os.path.join(snapshot_dir, "final_state.pkl")
            if not os.path.exists(final_snapshot_file):
                raise FileNotFoundError(f"找不到最终快照文件: {final_snapshot_file}")
            
            # 处理可能的numpy版本不兼容问题
            try:
                # 尝试直接加载
                with open(final_snapshot_file, 'rb') as f:
                    final_snapshot = pickle.load(f)
            except (ModuleNotFoundError, ImportError) as e:
                # 如果是numpy版本问题，打印警告并使用一个更简单的方法创建一个空环境
                self.status_label.config(text=f"警告：无法加载快照文件，创建模拟环境代替。错误: {str(e)}")
                import sys
                # 导入Environment类
                sys.path.append(project_root)
                from src.core.environment import Environment
                
                # 创建一个空的环境代替
                env = Environment(verbose=False)
                env.GRID_SIZE = env_config.get('grid_size', 20)
                
                # 创建一个简单的快照代替
                final_snapshot = {"env": env, "time_step": 0, "success_rate": 0}
            
            # 获取环境对象
            env = final_snapshot.get("env")
            if not env:
                raise ValueError("快照文件中没有找到环境对象")
            
            # 创建新的可视化框架
            viz_frame = ttk.Frame(self.env_frame)
            viz_frame.pack(fill=tk.BOTH, expand=True)
            
            # 创建环境快照列表：确保包含所有时间步骤
            env_snapshots = []
            
            # 收集进度数据以绘制成功率曲线
            progress_data = []
            for entry in metadata_entries:
                if 'time_step' in entry and 'success_rate' in entry:
                    progress_data.append((entry['time_step'], entry['success_rate']))
            
            # 确保按时间步排序
            metadata_entries.sort(key=lambda x: x.get('time_step', 0))
            
            # 加载每个step的环境快照
            for entry in metadata_entries:
                step_num = entry.get('step', 0)
                snapshot_file = os.path.join(snapshot_dir, f"step_{step_num:04d}.pkl")
                
                try:
                    with open(snapshot_file, 'rb') as f:
                        snapshot = pickle.load(f)
                        env_snapshots.append(snapshot)
                except (FileNotFoundError, pickle.UnpicklingError) as e:
                    print(f"警告：无法加载第 {step_num} 步的快照，使用元数据创建: {str(e)}")
                    env_copy = env.copy() if hasattr(env, 'copy') else env
                    snapshot_entry = {
                        "env": env_copy,
                        "time_step": entry.get('time_step', 0),
                        "success_rate": entry.get('success_rate', 0),
                        "disaster_count": entry.get('disaster_count', 0),
                        "success_count": entry.get('success_count', 0),
                        "total_reward": entry.get('total_reward', 0),
                        "avg_reward": entry.get('avg_reward', 0),
                        "avg_response_time": entry.get('avg_response_time', 0),
                        "episode": episode_num,
                        "step": step_num
                    }
                    env_snapshots.append(snapshot_entry)
            
            # 确保至少有一个快照
            if not env_snapshots:
                env_snapshots = [final_snapshot]
            
            # 确保快照按时间步排序
            env_snapshots.sort(key=lambda x: x.get('time_step', 0))
            
            # 更新状态标签
            self.status_label.config(text=f"已加载 {len(env_snapshots)} 个时间步的完全交互式可视化")
            
            # 处理警告
            import warnings
            warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
            warnings.filterwarnings("ignore", message="frames=None")
            
            # 在调用visualize前，先刷新事件队列
            self.root.update()
            
            # 直接调用visualize函数，获取图形对象
            fig = visualize(
                env_snapshots=env_snapshots,
                progress_data=progress_data,  # 使用正确格式的progress_data
                embedded_mode=True  # 确保返回图对象而不是显示
            )
            
            # 保存对象供后续使用
            self.current_fig = fig
            
            # 创建容器框架以提供更好的布局控制
            container_frame = ttk.Frame(viz_frame)
            container_frame.pack(fill=tk.BOTH, expand=True)
            
            # 创建FigureCanvasTkAgg对象并配置
            canvas = FigureCanvasTkAgg(fig, master=container_frame)
            
            # 显式设置图形大小以适应窗口
            fig.set_size_inches(10, 8)
            canvas.draw()
            
            # 保存对象供后续使用
            self.current_canvas = canvas
            
            # 将画布放置在框架中并配置以填充整个区域
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # 确保当窗口调整大小时，画布也会调整
            def on_resize(event):
                try:
                    # 根据窗口大小调整画布大小
                    width, height = event.width, event.height
                    if width > 100 and height > 100:  # 避免太小的尺寸
                        fig.set_size_inches(width/100, height/100)
                        canvas.draw_idle()
                except:
                    pass
            
            canvas_widget.bind("<Configure>", on_resize)
            
            # 添加导航工具栏
            toolbar_frame = ttk.Frame(container_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # 创建状态显示框架
            status_frame = ttk.Frame(viz_frame)
            status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
            
            # 添加坐标显示标签
            coord_label = ttk.Label(status_frame, text="坐标: ---, ---")
            coord_label.pack(side=tk.LEFT, padx=10)
            
            # 设置鼠标移动时的坐标显示
            def update_coord_status(event):
                if event.inaxes:
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:
                        coord_label.config(text=f"坐标: {x:.1f}, {y:.1f}")
            
            # 连接鼠标移动事件
            cid_motion = canvas.mpl_connect('motion_notify_event', update_coord_status)
            
            # 确保所有Matplotlib事件能正确传递
            def on_key_press(event):
                key_press_handler(event, canvas, toolbar)
            
            # 连接键盘事件
            cid_key = canvas.mpl_connect("key_press_event", on_key_press)
            
            # 处理鼠标事件特别是对滑动条的控制
            def on_button_press(event):
                # 记录事件已被处理以防止重复
                event.guiEvent.widget.focus_set()
            
            # 连接按钮按下事件
            cid_button = canvas.mpl_connect("button_press_event", on_button_press)
            
            # 清除任何可能存在的鼠标抓取冲突
            for ax in fig.get_axes():
                try:
                    # 处理可能的mousegrab冲突
                    if hasattr(ax, '_mousegrab_id') and ax._mousegrab_id:
                        try:
                            canvas.mpl_disconnect(ax._mousegrab_id)
                            ax._mousegrab_id = None
                        except:
                            pass
                    
                    # 防止重复绑定事件
                    if hasattr(ax, '_button_press'):
                        ax._button_press = None
                    if hasattr(ax, '_button_release'):
                        ax._button_release = None
                    if hasattr(ax, '_scroll_event'):
                        ax._scroll_event = None
                except:
                    pass
            
            # 当标签页改变或应用关闭时清理资源
            def on_window_close():
                """清理所有资源"""
                try:
                    # 停止所有定时器
                    if hasattr(self, 'update_timer_id') and self.update_timer_id:
                        self.root.after_cancel(self.update_timer_id)
                        self.update_timer_id = None
                    
                    # 关闭所有matplotlib图表
                    plt.close('all')
                    
                    # 清理画布
                    if hasattr(self, 'current_canvas') and self.current_canvas:
                        self.current_canvas.get_tk_widget().destroy()
                        self.current_canvas = None
                    
                    # 清理图形对象
                    if hasattr(self, 'current_fig') and self.current_fig:
                        self.current_fig = None
                    
                    # 清理环境框架中的所有部件
                    for widget in self.env_frame.winfo_children():
                        widget.destroy()
                except:
                    pass
            
            # 绑定标签页改变事件
            self.viz_notebook.bind("<<NotebookTabChanged>>", lambda e: on_window_close() if self.viz_notebook.select() != self.env_tab else None)
            
            # 当标签页改变或应用关闭时清理资源
            def on_cleanup():
                """清理所有资源"""
                try:
                    on_window_close() # 调用清理函数
                except:
                    pass
            
            # 创建轻量级的事件处理器以保持界面响应性
            # 这个函数每50毫秒处理一次matplotlib事件，确保滑动条和按钮能够正常工作
            def process_matplotlib_events():
                try:
                    # 处理Matplotlib事件（确保动画和滑动条正常工作）
                    if hasattr(fig, 'canvas') and fig.canvas:
                        # 刷新事件但不重绘整个画布
                        try:
                            fig.canvas.flush_events()
                        except:
                            pass
                
                    # 继续循环
                    self.update_timer_id = self.root.after(40, process_matplotlib_events)
                except Exception as e:
                    # 出错时打印信息但继续尝试
                    print(f"处理Matplotlib事件时出错: {e}")
                    # 尝试恢复
                    self.update_timer_id = self.root.after(100, process_matplotlib_events)
            
            # 启动事件处理循环（更频繁的更新以确保响应性）
            self.update_timer_id = self.root.after(40, process_matplotlib_events)
            
            # 更新状态标签
            self.status_label.config(text=f"已加载 {len(env_snapshots)} 个时间步的完全交互式可视化")
            
            # 切换到环境可视化标签页
            self.viz_notebook.select(self.env_tab)
            
        except Exception as e:
            # 记录错误并显示
            self.status_label.config(text=f"可视化出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def export_figures(self):
        """Export figures to files"""
        if not hasattr(self, 'selected_file'):
            return
            
        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select Directory to Save Figures")
        if not save_dir:
            return
            
        try:
            # Load metadata
            with open(self.selected_file, 'r') as f:
                data = json.load(f)
                
            # Get metrics
            metrics = data.get('metrics', {})
            if not metrics:
                raise ValueError("No metrics data found")
                
            # Create figures directory
            figures_dir = os.path.join(save_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
                
            # Plot and save each metric
            # Rewards
            if 'rewards' in metrics and metrics['rewards']:
                plt.figure(figsize=(8, 6))
                plt.plot(metrics['rewards'], '-b')
                plt.title('Average Reward')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.grid(True)
                plt.savefig(os.path.join(figures_dir, "rewards.png"))
                plt.close()
                
            # Success rates
            if 'success_rates' in metrics and metrics['success_rates']:
                plt.figure(figsize=(8, 6))
                plt.plot(metrics['success_rates'], '-g')
                plt.title('Rescue Success Rate')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate')
                plt.grid(True)
                plt.savefig(os.path.join(figures_dir, "success_rates.png"))
                plt.close()
                
            # Response times
            if 'response_times' in metrics and metrics['response_times']:
                plt.figure(figsize=(8, 6))
                plt.plot(metrics['response_times'], '-r')
                plt.title('Average Response Time')
                plt.xlabel('Episode')
                plt.ylabel('Seconds')
                plt.grid(True)
                plt.savefig(os.path.join(figures_dir, "response_times.png"))
                plt.close()
                
            # Losses
            if 'losses' in metrics and metrics['losses']:
                plt.figure(figsize=(8, 6))
                plt.plot(metrics['losses'], '-m')
                plt.title('Training Loss')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(os.path.join(figures_dir, "losses.png"))
                plt.close()
                
            self.status_label.config(text=f"Figures exported to {figures_dir}")
                
        except Exception as e:
            raise Exception(f"Error exporting figures: {str(e)}")

def main():
    """Start the visualization browser"""
    root = tk.Tk()
    app = TrainingDataBrowser(root)
    root.mainloop()

if __name__ == "__main__":
    main() 