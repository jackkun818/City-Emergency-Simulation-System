#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
城市应急救援模拟系统版本信息
"""

__version__ = "1.0.0"
__author__ = "City Emergency Team"
__email__ = "example@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2023 City Emergency Team"
__status__ = "Beta"
__date__ = "2023-11-01"

VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "beta",
}

def get_version():
    """返回当前版本字符串"""
    return __version__

def get_version_info():
    """返回版本信息字典"""
    return VERSION_INFO.copy()

def show_version_info():
    """打印版本信息"""
    print(f"城市应急救援模拟系统 v{__version__}")
    print(f"版本状态: {__status__}")
    print(f"发布日期: {__date__}")
    print(f"许可证: {__license__}")
    print(f"版权所有: {__copyright__}")

if __name__ == "__main__":
    show_version_info() 