@echo off
echo 开始安装城市应急救援模拟系统...

REM 检查Python版本
python -c "import sys; print('检测到Python版本:', '.'.join(map(str, sys.version_info[:2])))"

REM 创建虚拟环境
echo 创建虚拟环境...
python -m venv venv
call venv\Scripts\activate.bat

REM 升级pip
echo 升级pip...
python -m pip install --upgrade pip

REM 安装依赖
echo 安装项目依赖...
pip install -r requirements.txt

REM 开发模式安装
echo 以开发模式安装项目...
pip install -e .

REM 检查CUDA是否可用
echo 检查CUDA可用性...
python -c "import torch; print(f'CUDA是否可用: {torch.cuda.is_available()}')"

echo 安装完成！
echo 使用方法:
echo 1. 激活虚拟环境: venv\Scripts\activate.bat
echo 2. 训练模型: python train.py
echo 3. 运行模拟: python src\main.py

pause 