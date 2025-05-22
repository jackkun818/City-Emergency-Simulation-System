# 城市应急救援模拟系统

基于多智能体强化学习的城市应急救援模拟系统，用于优化救援资源分配和提高灾难响应效率。

## 项目简介

本系统模拟城市中的灾难事件和救援过程，使用多智能体强化学习（MARL）技术训练救援人员智能体，以提高救援效率和成功率。系统支持不同规模的灾难场景，并可以可视化救援过程。

### 主要特点

- 动态灾难生成与演化
- 多智能体强化学习训练框架
- 灵活的救援人员配置
- 多种任务分配算法（传统算法、MARL、混合策略）
- 训练过程可视化
- 性能指标追踪（救援成功率、平均响应时间）

## 安装指南

### 环境要求

- Python 3.7+
- CUDA支持（可选，用于GPU加速）

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/City-Emergency-Simulation-System.git
cd City-Emergency-Simulation-System
```

2. 安装依赖：

```bash
# 使用pip直接安装依赖
pip install -r requirements.txt

# 或者安装为开发模式
pip install -e .
```

## 使用方法

### 训练模型

```bash
# 使用默认参数训练
python train.py

# 自定义训练参数
python train.py --episodes 100 --steps 200 --save-freq 10
```

### 参数说明

- `--episodes`: 训练轮数
- `--steps`: 每轮训练步数
- `--save-freq`: 模型保存频率（每多少轮保存一次）

### 运行模拟

```bash
# 运行GUI模拟
python src/main.py

# 选择不同的任务分配算法
# 0: 传统混合算法
# 1: 多智能体强化学习
# 2: 混合算法
python src/main.py --algorithm 1
```

## 项目结构

```
City-Emergency-Simulation-System/
├── src/                    # 源代码
│   ├── core/               # 核心模拟逻辑
│   ├── rl/                 # 强化学习模块
│   ├── visualization/      # 可视化组件
│   └── main.py             # 主程序入口
├── train.py                # 训练脚本
├── requirements.txt        # 依赖库列表
├── setup.py                # 安装配置
└── README.md               # 项目说明
```

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。 