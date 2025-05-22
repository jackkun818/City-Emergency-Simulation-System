#!/bin/bash

# 城市应急救援模拟系统安装脚本
echo "开始安装城市应急救援模拟系统..."

# 检查Python版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "检测到Python版本: $python_version"

# 检查最低版本要求
if [[ $(echo "$python_version < 3.7" | bc) -eq 1 ]]; then
    echo "错误: 需要Python 3.7或更高版本"
    exit 1
fi

# 创建并激活虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 开发模式安装
echo "以开发模式安装项目..."
pip install -e .

# 检查CUDA是否可用
echo "检查CUDA可用性..."
python -c "import torch; print(f'CUDA是否可用: {torch.cuda.is_available()}')"

echo "安装完成！"
echo "使用方法:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 训练模型: python train.py"
echo "3. 运行模拟: python src/main.py" 