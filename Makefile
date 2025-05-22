.PHONY: install install-dev test train run clean

# 默认目标
all: install

# 安装依赖
install:
	pip install -r requirements.txt

# 开发模式安装
install-dev:
	pip install -e .

# 运行测试
test:
	python -m unittest discover -s tests

# 训练模型
train:
	python train.py

# 自定义训练参数
train-custom:
	python train.py --episodes 100 --steps 200 --save-freq 10

# 运行模拟
run:
	python src/main.py

# 使用特定算法运行模拟
# 0: 传统混合算法
# 1: 多智能体强化学习
# 2: 混合算法
run-marl:
	python src/main.py --algorithm 1

# 清理生成的文件
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

# 创建虚拟环境
venv:
	python -m venv venv
	@echo "使用 'source venv/bin/activate' (Linux/Mac) 或 'venv\\Scripts\\activate' (Windows) 激活虚拟环境"

# 帮助信息
help:
	@echo "可用命令:"
	@echo "  make install      - 安装依赖"
	@echo "  make install-dev  - 开发模式安装"
	@echo "  make venv         - 创建虚拟环境"
	@echo "  make test         - 运行测试"
	@echo "  make train        - 训练模型"
	@echo "  make train-custom - 使用自定义参数训练模型"
	@echo "  make run          - 运行模拟"
	@echo "  make run-marl     - 使用MARL算法运行模拟"
	@echo "  make clean        - 清理生成的文件" 