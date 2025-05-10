#!/usr/bin/env python3
"""
训练数据可视化浏览器启动脚本
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

def main():
    """启动可视化浏览器"""
    # 尝试导入可视化浏览器模块
    try:
        from src.visualization.viz_browser import main as start_browser
        # 启动浏览器
        start_browser()
    except ImportError as e:
        print(f"导入可视化工具失败: {e}")
        print("请确保您已安装所需的依赖包:")
        print("  - tkinter")
        print("  - matplotlib")
        print("  - numpy")
        print("  - pillow")
    except Exception as e:
        print(f"启动可视化工具时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 