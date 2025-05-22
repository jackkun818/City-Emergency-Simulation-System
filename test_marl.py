#!/usr/bin/env python3
"""
MARL系统测试脚本 - 测试救援人员参数保存和加载

用法:
  python test_marl.py
"""

import os
import sys
import json

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.environment import Environment
from src.rl.marl_rescue import RescueEnvironment, MARLController
from src.rl.marl_integration import initialize_marl_controller
from src.core import config

def test_save_load_rescuers():
    print("===== 测试救援人员参数保存和加载 =====")
    
    # 1. 创建环境和初始救援人员
    print("\n[1] 创建初始环境和救援人员")
    env = Environment(verbose=True)
    initial_rescuers = env.rescuers
    print(f"初始救援人员数量: {len(initial_rescuers)}")
    
    # 打印救援人员信息
    for i, rescuer in enumerate(initial_rescuers):
        print(f"  救援人员 {i+1}: 位置={rescuer['position']}, 能力={rescuer['capacity']}, 速度={rescuer['speed']}")
    
    # 2. 保存救援人员数据
    print("\n[2] 保存救援人员数据")
    temp_env = RescueEnvironment(grid_size=env.GRID_SIZE, rescuers_data=initial_rescuers)
    model_dir = os.path.join(os.path.dirname(config.MARL_CONFIG['model_save_path']))
    rescuers_data_path = os.path.join(model_dir, "rescuers_data.json")
    os.makedirs(model_dir, exist_ok=True)
    temp_env.save_rescuers_data(rescuers_data_path)
    print(f"已保存救援人员数据到: {rescuers_data_path}")
    
    # 3. 加载救援人员数据
    print("\n[3] 加载救援人员数据")
    loaded_rescuers = RescueEnvironment.load_rescuers_data(rescuers_data_path)
    print(f"已加载救援人员数据，数量: {len(loaded_rescuers)}")
    
    # 打印加载的救援人员信息
    for i, rescuer in enumerate(loaded_rescuers):
        print(f"  救援人员 {i+1}: 位置={rescuer['position']}, 能力={rescuer.get('capacity', 'N/A')}, 速度={rescuer.get('speed', 'N/A')}")
    
    # 4. 验证救援人员参数是否一致
    print("\n[4] 验证救援人员参数是否一致")
    all_match = True
    for i, (original, loaded) in enumerate(zip(initial_rescuers, loaded_rescuers)):
        match = (
            original['id'] == loaded['id'] and
            original['position'] == loaded['position'] and
            original['capacity'] == loaded.get('capacity', None) and
            original['speed'] == loaded.get('speed', None)
        )
        if not match:
            all_match = False
            print(f"  救援人员 {i+1} 参数不匹配!")
            print(f"    原始: ID={original['id']}, 位置={original['position']}, 能力={original['capacity']}, 速度={original['speed']}")
            print(f"    加载: ID={loaded['id']}, 位置={loaded['position']}, 能力={loaded.get('capacity', 'N/A')}, 速度={loaded.get('speed', 'N/A')}")
    
    if all_match:
        print("  所有救援人员参数完全匹配！")
    
    # 5. 测试通过部署API加载救援人员
    print("\n[5] 测试通过部署API加载救援人员")
    new_env = Environment(verbose=False)
    controller = initialize_marl_controller(new_env, use_saved_rescuers=True)
    
    # 6. 验证部署环境中的救援人员参数
    print("\n[6] 验证部署环境中的救援人员参数")
    deployed_rescuers = new_env.rescuers
    all_match = True
    for i, (original, deployed) in enumerate(zip(initial_rescuers, deployed_rescuers)):
        match = (
            original['id'] == deployed['id'] and
            original['position'] == deployed['position'] and
            original['capacity'] == deployed.get('capacity', None) and
            original['speed'] == deployed.get('speed', None)
        )
        if not match:
            all_match = False
            print(f"  部署救援人员 {i+1} 参数不匹配!")
            print(f"    原始: ID={original['id']}, 位置={original['position']}, 能力={original['capacity']}, 速度={original['speed']}")
            print(f"    部署: ID={deployed['id']}, 位置={deployed['position']}, 能力={deployed.get('capacity', 'N/A')}, 速度={deployed.get('speed', 'N/A')}")
    
    if all_match:
        print("  部署环境中的救援人员参数完全匹配！")
    
    print("\n===== 测试完成 =====")
    if all_match:
        print("测试结果: 成功 ✓")
    else:
        print("测试结果: 失败 ✗")

if __name__ == "__main__":
    test_save_load_rescuers() 