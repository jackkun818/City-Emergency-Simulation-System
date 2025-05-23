from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
import sys, os
import socket

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import webbrowser
from threading import Timer
from src.visualization.visualization import visualize
from src.core.environment import Environment
from src.core.rescue_execution import execute_rescue
from src.rl.marl_integration import dispatch_rescue_tasks
from src.core import config
import copy
from src.utils.stats import calculate_rescue_success_rate

import mpld3
from mpld3 import plugins

from src.visualization.visualization import export_visualization_video

# 确保Flask能找到模板和静态文件目录
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# 默认参数（可以被前端修改）
SIMULATION_PARAMS = {
    "NUM_RESCUERS": 10,
    "MAX_RESCUE_CAPACITY": 3,
    "MAX_SPEED": 3
}

# 模拟环境快照缓存和进度数据
env_snapshots = []
progress_data = []


# 包装 visualize 函数用于网页输出
def generate_visualization_html():
    if not env_snapshots:
        return "<p>No simulation data available. Please run the simulation first.</p>"

    # 设置matplotlib后端为Agg
    plt.switch_backend('Agg')

    # 创建图形
    fig = visualize(env_snapshots, progress_data)

    # 添加交互式功能
    plugins.connect(fig, plugins.MousePosition())

    # 转换为HTML，包含所有必要的JavaScript
    html_str = mpld3.fig_to_html(fig, template_type='simple')
    plt.close(fig)
    return html_str


@app.route('/')
def index():
    return render_template('index.html', params=SIMULATION_PARAMS)


@app.route('/start', methods=['POST'])
def start_simulation():
    # 获取用户输入参数
    num_rescuers = int(request.form.get("num_rescuers", 10))
    capacity = int(request.form.get("capacity", 3))
    speed = int(request.form.get("speed", 3))

    # 更新全局参数
    SIMULATION_PARAMS.update({
        "NUM_RESCUERS": num_rescuers,
        "MAX_RESCUE_CAPACITY": capacity,
        "MAX_SPEED": speed
    })

    # 设置 config 全局参数
    config.NUM_RESCUERS = num_rescuers
    config.MAX_RESCUE_CAPACITY = capacity
    config.MAX_SPEED = speed

    # 构建新环境并生成快照
    env = Environment()
    global env_snapshots, progress_data
    env_snapshots = []
    progress_data = []

    for t in range(30):
        env.update_disasters(current_time_step=t)
        dispatch_rescue_tasks(env.rescuers, env.disasters, env.GRID_SIZE, current_time_step=t)
        execute_rescue(env.rescuers, env.disasters, env.GRID_SIZE, current_time_step=t)

        # 计算当前时间步的成功率
        success_rate = calculate_rescue_success_rate(env.disasters, window=30, current_time_step=t)
        progress_data.append((t, success_rate))

        # 保存环境快照
        env_snapshots.append({
            "env": copy.deepcopy(env),
            "time_step": t,
            "success_rate": success_rate
        })

    return jsonify({"status": "simulation completed"})


@app.route('/plot', methods=['POST'])
def plot():
    try:
        export_visualization_video(env_snapshots, progress_data, output_path="static/visualization.mp4")
        return jsonify({"video": "/static/visualization.mp4"})
    except Exception as e:
        print(f"视频生成错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    global env_snapshots, progress_data
    SIMULATION_PARAMS.update({
        "NUM_RESCUERS": 10,
        "MAX_RESCUE_CAPACITY": 3,
        "MAX_SPEED": 3
    })
    env_snapshots = []
    progress_data = []
    return jsonify({"status": "reset"})


def open_browser():
    webbrowser.open_new(f'http://127.0.0.1:{port}/')


def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == '__main__':
    port = 5000
    # 如果5000端口被占用，尝试5001端口
    while (is_port_in_use(port)):
        port += 1
        
    print(f"启动服务器，使用端口: {port}")
    Timer(1, open_browser).start()
    app.run(debug=True, port=port)

