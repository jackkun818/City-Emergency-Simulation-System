# 城市应急救援模拟系统安装指南

本文档提供了多种安装城市应急救援模拟系统的方法，适用于不同的操作系统和用户偏好。

## 目录

- [环境要求](#环境要求)
- [方法1：使用自动安装脚本](#方法1使用自动安装脚本)
- [方法2：使用pip安装](#方法2使用pip安装)
- [方法3：使用Conda安装](#方法3使用conda安装)
- [方法4：使用Make命令安装](#方法4使用make命令安装)
- [GPU支持](#gpu支持)
- [常见问题](#常见问题)

## 环境要求

- Python 3.7+
- 足够的磁盘空间（至少1GB）
- 推荐：CUDA兼容的GPU（用于加速训练）

## 方法1：使用自动安装脚本

### Linux/macOS用户

```bash
# 克隆仓库
git clone https://github.com/yourusername/City-Emergency-Simulation-System.git
cd City-Emergency-Simulation-System

# 添加执行权限
chmod +x install.sh

# 运行安装脚本
./install.sh
```

### Windows用户

```bash
# 克隆仓库
git clone https://github.com/yourusername/City-Emergency-Simulation-System.git
cd City-Emergency-Simulation-System

# 运行安装脚本
install.bat
```

## 方法2：使用pip安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/City-Emergency-Simulation-System.git
cd City-Emergency-Simulation-System

# 创建并激活虚拟环境（可选但推荐）
python -m venv venv
# 在Linux/macOS上:
source venv/bin/activate
# 在Windows上:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 开发模式安装（可选）
pip install -e .
```

## 方法3：使用Conda安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/City-Emergency-Simulation-System.git
cd City-Emergency-Simulation-System

# 创建并激活Conda环境
conda env create -f environment.yml
conda activate city-emergency

# 验证安装
python -c "import torch; print(f'CUDA是否可用: {torch.cuda.is_available()}')"
```

## 方法4：使用Make命令安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/City-Emergency-Simulation-System.git
cd City-Emergency-Simulation-System

# 创建虚拟环境
make venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
make install

# 开发模式安装（可选）
make install-dev
```

## GPU支持

为了获得最佳性能，建议使用支持CUDA的GPU。安装适合您系统的PyTorch CUDA版本：

```bash
# 例如，安装支持CUDA 11.3的PyTorch
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## 常见问题

### 问题：安装时出现"ERROR: Could not find a version that satisfies the requirement..."

**解决方案**：尝试升级pip并重新安装

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 问题：导入torch时出现错误

**解决方案**：确保安装了正确版本的PyTorch

```bash
pip uninstall torch
pip install torch>=1.10.0
```

### 问题：无法使用GPU加速

**解决方案**：检查CUDA安装和PyTorch版本兼容性

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

如果您遇到其他问题，请在GitHub仓库中提交issue。 