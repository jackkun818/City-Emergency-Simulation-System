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
from matplotlib.backend_bases import key_press_handler  # 导入key_press_handler函数
import numpy as np
import sys
import re

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import visualization module
from .visualization import visualize

# Set save directory
SAVE_DIR = os.path.join(project_root, 'train_visualization_save')

# Default data directory
DEFAULT_DATA_DIR = os.path.join(SAVE_DIR, 'metadata')

class TrainingDataBrowser:
    def __init__(self, root, data_dir=DEFAULT_DATA_DIR):
        self.root = root
        self.root.title("Training Data Visualization Browser")
        self.root.geometry("1200x800")
        self.data_dir = data_dir
        
        # Create main interface
        self.setup_ui()
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            # Create data directory
            parent_dir = os.path.dirname(self.data_dir)
            if os.path.exists(parent_dir):
                os.makedirs(self.data_dir, exist_ok=True)
                self.status_label.config(text=f"Created data directory: {self.data_dir}")
            else:
                # If the parent directory doesn't exist, prompt the user to select a valid directory
                self.status_label.config(text="Default data directory doesn't exist. Please select a valid directory.")
                # Delay call to choose_data_dir to ensure UI is fully loaded
                self.root.after(100, self.choose_data_dir)
                return
        
        # Load metadata file list
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
    
    def on_tab_changed(self, event):
        """Handle tab selection event"""
        if not hasattr(self, 'selected_file'):
            return
            
        selected_tab = self.viz_notebook.select()
        tab_index = self.viz_notebook.index(selected_tab)
        
        # 根据标签页索引执行相应操作
        try:
            # 基本信息标签页
            if tab_index == 0:
                self.show_basic_info(self.selected_file)
            
            # 训练指标标签页
            elif tab_index == 1:
                with open(self.selected_file, 'r') as f:
                    data = json.load(f)
                metrics = data.get('metrics', {})
                if metrics:
                    self.plot_metrics(metrics)
            
            # 环境可视化标签页
            elif tab_index == 2:
                # 避免重复加载，如果可视化区域已经有内容，不重新加载
                if len(self.env_frame.winfo_children()) == 0:
                    self.visualize_selected()
        except Exception as e:
            # 捕获错误并显示在状态栏，但不中断
            self.status_label.config(text=f"Error when switching tabs: {str(e)}")
    
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
        """File selection event handler"""
        selected_items = self.file_tree.selection()
        
        if not selected_items:
            self.visualize_btn.config(state=tk.DISABLED)
            return
        
        # Enable visualization button
        self.visualize_btn.config(state=tk.NORMAL)
        
        # Get selected file name
        item = selected_items[0]
        file_tag = self.file_tree.item(item, "tags")[0]
        
        # Store selected file path
        self.selected_file = os.path.join(self.data_dir, file_tag)
        
        # Show basic information
        self.show_basic_info(self.selected_file)
        
        # Load and plot metrics
        with open(self.selected_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('metrics', {})
        if metrics:
            self.plot_metrics(metrics)
            self.export_btn.config(state=tk.NORMAL)
        else:
            self.export_btn.config(state=tk.DISABLED)
        
        # 立即尝试加载环境可视化
        try:
            self.visualize_selected()
        except Exception as e:
            # 只记录错误，不中断
            self.status_label.config(text=f"Error loading visualization: {str(e)}")
        
        # Update status
        self.status_label.config(text=f"Selected file: {file_tag}")
    
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
            if 'rewards' in metrics and metrics['rewards']:
                self.info_text.insert(tk.END, f"• Final Reward: {metrics['rewards'][-1]:.4f}\n")
            if 'success_rates' in metrics and metrics['success_rates']:
                self.info_text.insert(tk.END, f"• Final Success Rate: {metrics['success_rates'][-1]:.4f}\n")
            if 'response_times' in metrics and metrics['response_times']:
                self.info_text.insert(tk.END, f"• Final Response Time: {metrics['response_times'][-1]:.4f}\n")
            if 'losses' in metrics and metrics['losses']:
                self.info_text.insert(tk.END, f"• Final Loss: {metrics['losses'][-1]:.4f}\n")
            
        except Exception as e:
            self.info_text.insert(tk.END, f"Error loading file: {str(e)}")
    
    def plot_metrics(self, metrics):
        """
        Plot training metrics
        
        The training metrics include:
        - Rewards: Average reward per episode, indicates overall agent performance
        - Success Rates: Percentage of successful rescues, key indicator of effectiveness
        - Response Times: Average time taken to respond to disasters, lower is better
        - Losses: Training loss values, indicates model convergence
        """
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
        """为选中的训练数据生成并播放重复的训练step过程视频"""
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
            
            # 确保按时间步排序
            metadata_entries.sort(key=lambda x: x.get('step', 0))
            
            # 加载每个步骤的真实环境快照
            env_snapshots = []
            
            self.status_label.config(text="正在加载环境快照...")
            self.root.update()
            
            # 为每个元数据条目加载对应的环境快照
            for i, entry in enumerate(metadata_entries):
                step_num = entry.get('step', i + 1)
                
                # 构建步骤快照文件路径
                step_snapshot_file = os.path.join(snapshot_dir, f"step_{step_num:04d}.pkl")
                
                # 尝试加载步骤快照
                snapshot_loaded = False
                if os.path.exists(step_snapshot_file):
                    try:
                        with open(step_snapshot_file, 'rb') as f:
                            snapshot = pickle.load(f)
                        
                        # 验证快照数据的完整性
                        if isinstance(snapshot, dict) and "env" in snapshot:
                            # 补充元数据信息（以防快照中缺少某些字段）
                            snapshot.update({
                                "time_step": entry.get('time_step', step_num),
                                "success_rate": entry.get('success_rate', snapshot.get('success_rate', 0)),
                                "disaster_count": entry.get('disaster_count', snapshot.get('disaster_count', 0)),
                                "success_count": entry.get('success_count', snapshot.get('success_count', 0)),
                                "total_reward": entry.get('total_reward', snapshot.get('total_reward', 0)),
                                "avg_reward": entry.get('avg_reward', snapshot.get('avg_reward', 0)),
                                "avg_response_time": entry.get('avg_response_time', snapshot.get('avg_response_time', 0)),
                                "episode": episode_num,
                                "step": step_num
                            })
                            
                            env_snapshots.append(snapshot)
                            snapshot_loaded = True
                            
                            # 每10步更新一次状态
                            if step_num % 10 == 0:
                                self.status_label.config(text=f"正在加载环境快照... {step_num}/{len(metadata_entries)}")
                                self.root.update()
                    
                    except Exception as e:
                        print(f"警告：无法加载步骤 {step_num} 的快照文件 {step_snapshot_file}: {e}")
                
                # 如果无法加载步骤快照，尝试使用最终快照作为备选
                if not snapshot_loaded:
                    print(f"警告：步骤 {step_num} 快照文件不存在或损坏，将尝试使用最终快照")
                    
                    final_snapshot_file = os.path.join(snapshot_dir, "final_state.pkl")
                    if os.path.exists(final_snapshot_file):
                        try:
                            with open(final_snapshot_file, 'rb') as f:
                                final_snapshot = pickle.load(f)
                            
                            # 创建一个基于最终快照的副本，但使用当前步骤的元数据
                            snapshot_entry = {
                                "env": final_snapshot.get("env"),
                                "time_step": entry.get('time_step', step_num),
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
                            snapshot_loaded = True
                        except Exception as e:
                            print(f"错误：无法加载最终快照作为备选: {e}")
                
                # 如果都无法加载，跳过这个步骤
                if not snapshot_loaded:
                    print(f"警告：跳过步骤 {step_num}，无法加载任何有效快照")
            
            # 检查是否成功加载了快照
            if not env_snapshots:
                raise ValueError("无法加载任何有效的环境快照。请检查快照文件是否存在且未损坏。")
            
            # 收集进度数据以绘制成功率曲线
            progress_data = [(entry['time_step'], entry['success_rate']) for entry in env_snapshots]
            
            # 创建主容器框架
            main_frame = ttk.Frame(self.env_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 创建标题标签
            title_label = ttk.Label(main_frame, 
                                   text=f"训练Episode {episode_num} - 全程step动画播放", 
                                   font=("Arial", 14, "bold"))
            title_label.pack(pady=(0, 10))
            
            # 创建视频信息框架
            info_frame = ttk.Frame(main_frame)
            info_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 显示基本信息
            info_text = f"""
训练步骤总数: {len(env_snapshots)} steps
预计播放时长: 约 {len(env_snapshots) * 0.5:.1f} 秒 (每step 0.5秒)
播放模式: 自动循环播放
状态: 正在初始化动画...
            """.strip()
            
            self.info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
            self.info_label.pack(anchor=tk.W)
            
            # 创建控制按钮框架
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 播放控制变量
            self.is_playing = False
            self.current_step = 0
            self.animation_timer_id = None
            
            # 左侧按钮组
            left_buttons = ttk.Frame(control_frame)
            left_buttons.pack(side=tk.LEFT)
            
            # 播放/暂停按钮 - 固定大小
            self.play_pause_btn = ttk.Button(left_buttons, text="▶ 开始播放", 
                                           command=self.toggle_playback,
                                           width=12)
            self.play_pause_btn.pack(side=tk.LEFT, padx=(0, 5))
            
            # 停止按钮 - 固定大小
            self.stop_btn = ttk.Button(left_buttons, text="⏹ 停止", 
                                     command=self.stop_playback,
                                     width=8)
            self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            # 中间速度控制组
            speed_frame = ttk.Frame(control_frame)
            speed_frame.pack(side=tk.LEFT, padx=(10, 0))
            
            # 速度控制
            speed_label = ttk.Label(speed_frame, text="播放速度:")
            speed_label.pack(side=tk.LEFT, padx=(0, 5))
            
            self.speed_var = tk.StringVar(value="正常")
            speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, 
                                     values=["慢速", "正常", "快速"], width=8, state="readonly")
            speed_combo.pack(side=tk.LEFT)
            
            # 右侧步骤显示
            self.step_label = ttk.Label(control_frame, text="当前步骤: 0/0")
            self.step_label.pack(side=tk.RIGHT)
            
            # 创建matplotlib嵌入式画布
            viz_frame = ttk.LabelFrame(main_frame, text="训练步骤可视化 (循环播放)")
            viz_frame.pack(fill=tk.BOTH, expand=True)
            
            # 使用matplotlib后端创建动画
            matplotlib.use('TkAgg')
            
            # 导入可视化函数
            from .visualization import visualize
            
            # 创建图形对象 - 使用第一个快照初始化
            self.fig = visualize(
                env_snapshots=[env_snapshots[0]], 
                progress_data=[(0, env_snapshots[0].get('success_rate', 0))], 
                embedded_mode=True
            )
            
            # 存储环境快照以供动画使用
            self.env_snapshots = env_snapshots
            self.progress_data = progress_data
            
            # 创建画布
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 添加导航工具栏
            toolbar_frame = ttk.Frame(viz_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            toolbar.update()
            
            # 更新步骤显示
            self.step_label.config(text=f"当前步骤: 1/{len(env_snapshots)}")
            
            # 更新信息标签
            updated_info = f"""
训练步骤总数: {len(env_snapshots)} steps
预计播放时长: 约 {len(env_snapshots) * 0.5:.1f} 秒 (每step 0.5秒)
播放模式: 自动循环播放
状态: ✅ 动画已就绪，点击播放按钮开始
            """.strip()
            self.info_label.config(text=updated_info)
            
            # 更新主状态标签
            self.status_label.config(text=f"已加载 {len(env_snapshots)} 个真实训练步骤快照")
            
            # 切换到环境可视化标签页
            self.viz_notebook.select(self.env_tab)
            
        except Exception as e:
            # 记录错误并显示
            self.status_label.config(text=f"动画加载出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def toggle_playback(self):
        """切换播放/暂停状态"""
        if not hasattr(self, 'env_snapshots') or not self.env_snapshots:
            return
        
        if self.is_playing:
            # 当前正在播放，切换到暂停
            self.is_playing = False
            if self.animation_timer_id:
                self.root.after_cancel(self.animation_timer_id)
                self.animation_timer_id = None
            self.play_pause_btn.config(text="▶ 继续播放")
        else:
            # 当前暂停，开始播放
            self.is_playing = True
            self.play_pause_btn.config(text="⏸ 暂停播放")
            self.start_animation()
    
    def stop_playback(self):
        """停止播放并重置到第一步"""
        self.is_playing = False
        if self.animation_timer_id:
            self.root.after_cancel(self.animation_timer_id)
            self.animation_timer_id = None
        
        # 重置到第一步
        self.current_step = 0
        self.play_pause_btn.config(text="▶ 开始播放")
        
        # 更新显示
        if hasattr(self, 'env_snapshots') and self.env_snapshots:
            self.update_visualization()
            self.step_label.config(text=f"当前步骤: 1/{len(self.env_snapshots)}")
    
    def start_animation(self):
        """开始动画循环"""
        if not self.is_playing:
            return
        
        # 更新当前可视化
        self.update_visualization()
        
        # 计算播放间隔（基于速度设置）
        speed = self.speed_var.get()
        if speed == "慢速":
            interval = 1000  # 1秒
        elif speed == "快速":
            interval = 200   # 0.2秒
        else:  # 正常
            interval = 500   # 0.5秒
        
        # 移动到下一步
        self.current_step = (self.current_step + 1) % len(self.env_snapshots)
        
        # 更新步骤显示
        self.step_label.config(text=f"当前步骤: {self.current_step + 1}/{len(self.env_snapshots)}")
        
        # 安排下一次更新
        if self.is_playing:
            self.animation_timer_id = self.root.after(interval, self.start_animation)
    
    def update_visualization(self):
        """更新可视化内容到当前步骤"""
        if not hasattr(self, 'env_snapshots') or not self.env_snapshots:
            return
        
        if not hasattr(self, 'fig') or not self.fig:
            return
        
        try:
            # 获取当前步骤的快照
            current_snapshot = self.env_snapshots[self.current_step]
            
            # 获取当前步骤之前的累积进度数据
            cumulative_progress = [(entry['time_step'], entry['success_rate']) 
                                 for i, entry in enumerate(self.env_snapshots) 
                                 if i <= self.current_step]
            
            # 清除当前图形
            self.fig.clear()
            
            # 导入可视化函数并生成新的可视化
            from .visualization import visualize
            
            # 生成新的可视化图形
            new_fig = visualize(
                env_snapshots=[current_snapshot], 
                progress_data=cumulative_progress, 
                embedded_mode=True
            )
            
            # 将新图形的内容复制到现有图形
            self._copy_figure_content(new_fig, self.fig)
            
            # 关闭临时图形以释放内存
            plt.close(new_fig)
            
            # 调整布局并重新绘制
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"更新可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _copy_figure_content(self, source_fig, target_fig):
        """将源图形的内容复制到目标图形，保持visualize函数的原始布局"""
        try:
            # 获取源图形的所有轴
            source_axes = source_fig.get_axes()
            
            if len(source_axes) == 0:
                return
            
            # 清除目标图形的所有子图
            target_fig.clear()
            
            # 根据visualize函数的布局重新创建子图
            # 使用与原始visualize函数相同的subplot2grid布局
            # 网格图（上方，占70%）：plt.subplot2grid((10, 2), (0, 0), rowspan=7, colspan=2)
            grid_ax = plt.subplot2grid((10, 2), (0, 0), rowspan=7, colspan=2, fig=target_fig)
            # 成功率图（下方，占20%）：plt.subplot2grid((10, 2), (7, 0), rowspan=2, colspan=2)
            rate_ax = plt.subplot2grid((10, 2), (7, 0), rowspan=2, colspan=2, fig=target_fig)
            
            target_axes = [grid_ax, rate_ax]
            
            # 只复制前两个轴（网格图和成功率图），忽略滑动条
            source_axes_to_copy = source_axes[:2] if len(source_axes) >= 2 else source_axes
            
            # 复制每个轴的内容
            for i, (src_ax, tgt_ax) in enumerate(zip(source_axes_to_copy, target_axes)):
                # 复制基本的图形元素
                self._copy_axis_content(src_ax, tgt_ax)
                
                # 特殊处理：确保网格图的坐标轴设置正确
                if i == 0:  # 网格图
                    # 保持固定的坐标轴范围和网格设置
                    tgt_ax.set_xlim(src_ax.get_xlim())
                    tgt_ax.set_ylim(src_ax.get_ylim())
                    tgt_ax.grid(True)
                    # 确保刻度标签为空（因为是网格图）
                    tgt_ax.set_xticklabels([])
                    tgt_ax.set_yticklabels([])
                elif i == 1:  # 成功率图
                    # 确保成功率图的y轴范围正确
                    tgt_ax.set_ylim(0, 1.05)
                    tgt_ax.grid(True)
                
        except Exception as e:
            print(f"复制图形内容时出错: {e}")
            # 如果复制失败，显示错误信息
            target_fig.clear()
            error_ax = target_fig.add_subplot(111)
            error_ax.text(0.5, 0.5, f"可视化更新出错:\n{str(e)}", 
                         ha='center', va='center', transform=error_ax.transAxes,
                         fontsize=12, color='red')
    
    def _copy_axis_content(self, source_ax, target_ax):
        """
        将源轴的内容复制到目标轴
        """
        try:
            # 复制图像
            for image in source_ax.images:
                im_data = image.get_array()
                extent = image.get_extent()
                target_ax.imshow(im_data, extent=extent, aspect=image.get_aspect(),
                               interpolation=image.get_interpolation(), 
                               cmap=image.get_cmap(), alpha=image.get_alpha())
            
            # 复制线条
            for line in source_ax.lines:
                x_data, y_data = line.get_data()
                target_ax.plot(x_data, y_data, 
                             color=line.get_color(),
                             linewidth=line.get_linewidth(),
                             linestyle=line.get_linestyle(),
                             marker=line.get_marker(),
                             markersize=line.get_markersize(),
                             alpha=line.get_alpha())
            
            # 复制散点图
            for collection in source_ax.collections:
                if hasattr(collection, 'get_offsets'):
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        x_data = offsets[:, 0]
                        y_data = offsets[:, 1]
                        target_ax.scatter(x_data, y_data,
                                        s=collection.get_sizes(),
                                        c=collection.get_facecolors(),
                                        marker=collection.get_paths()[0] if collection.get_paths() else 'o',
                                        alpha=collection.get_alpha())
            
            # 复制文字
            for text in source_ax.texts:
                target_ax.text(text.get_position()[0], text.get_position()[1], 
                             text.get_text(),
                             fontsize=text.get_fontsize(),
                             color=text.get_color(),
                             alpha=text.get_alpha())
            
            # 复制标题和标签
            if source_ax.get_title():
                target_ax.set_title(source_ax.get_title())
            if source_ax.get_xlabel():
                target_ax.set_xlabel(source_ax.get_xlabel())
            if source_ax.get_ylabel():
                target_ax.set_ylabel(source_ax.get_ylabel())
            
            # 复制轴范围
            target_ax.set_xlim(source_ax.get_xlim())
            target_ax.set_ylim(source_ax.get_ylim())
            
            # 简化的网格设置复制 - 仅检查是否启用网格
            try:
                if source_ax.xaxis._major_tick_kw.get('gridOn', False):
                    target_ax.grid(True, axis='x')
                if source_ax.yaxis._major_tick_kw.get('gridOn', False):
                    target_ax.grid(True, axis='y')
            except (AttributeError, KeyError):
                # 如果无法获取网格状态，跳过网格设置
                pass
            
            # 复制图例 - 使用更简单的方法
            try:
                legend = source_ax.get_legend()
                if legend:
                    labels = [t.get_text() for t in legend.get_texts()]
                    if labels:
                        target_ax.legend(labels, loc='best')
            except Exception as e:
                print(f"跳过图例复制: {e}")
                
        except Exception as e:
            print(f"复制轴内容时出错: {e}")
    
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
    """Main function to start the application"""
    try:
        root = tk.Tk()
        app = TrainingDataBrowser(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 