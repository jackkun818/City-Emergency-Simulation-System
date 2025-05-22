#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# 读取README.md文件内容
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "城市应急救援多智能体强化学习模拟系统"

# 读取requirements.txt文件内容
try:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = [
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "torch>=1.10.0",
        "gymnasium>=0.28.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "pillow>=9.0.0"
    ]

# 获取版本号
version = "1.0.0"
version_file = os.path.join("src", "version.py")
if os.path.exists(version_file):
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

setup(
    name="city-emergency-simulation",
    version=version,
    author="City Emergency Team",
    author_email="example@example.com",
    description="城市应急救援多智能体强化学习模拟系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/City-Emergency-Simulation-System",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/City-Emergency-Simulation-System/issues",
        "Documentation": "https://github.com/yourusername/City-Emergency-Simulation-System/wiki",
        "Source Code": "https://github.com/yourusername/City-Emergency-Simulation-System",
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
            "isort>=5.7.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "city-simulation=train:main",
            "city-viz=src.visualization.visualization:main",
        ],
    },
) 