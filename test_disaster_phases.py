#!/usr/bin/env python3
"""
测试三阶段灾难生成策略的验证脚本
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.environment import Environment
from src.rl.rl_util import adjust_disaster_settings

def test_disaster_phases():
    """测试三阶段灾难策略是否正确工作"""
    print("🔍 测试三阶段灾难生成策略...")
    
    # 创建环境
    env = Environment(grid_size=20, num_rescuers=10, training_mode=True)
    max_steps = 100  # 使用较短的步数进行测试
    
    # 记录每个阶段的活跃灾难数
    phase_data = {"phase1": [], "phase2": [], "phase3": []}
    
    for step in range(1, max_steps + 1):
        # 调用灾难设置调整函数
        adjust_disaster_settings(env, step, max_steps, verbose=False)
        
        # 更新灾难状态
        env.update_disasters_silent(current_time_step=step)
        
        # 统计当前活跃灾难数
        active_disasters = sum(1 for d in env.disasters.values() if d.get("rescue_needed", 0) > 0)
        total_disasters = len(env.disasters)
        
        # 根据步数分配到不同阶段
        phase1_end = int(max_steps * 2 / 3)  # 67步
        phase2_end = int(max_steps * 5 / 6)  # 83步
        
        if step <= phase1_end:
            phase_data["phase1"].append(active_disasters)
            phase = "阶段1"
        elif step <= phase2_end:
            phase_data["phase2"].append(active_disasters)
            phase = "阶段2"
        else:
            phase_data["phase3"].append(active_disasters)
            phase = "阶段3"
        
        # 每10步输出一次状态
        if step % 10 == 0:
            print(f"步骤 {step:3d}/{max_steps} ({phase}): 活跃灾难={active_disasters:2d}, 总灾难={total_disasters:2d}")
    
    # 分析结果
    print("\n📊 三阶段灾难生成效果分析:")
    
    if phase_data["phase1"]:
        avg_phase1 = np.mean(phase_data["phase1"])
        print(f"阶段1 (步骤1-{phase1_end}): 平均活跃灾难数 = {avg_phase1:.1f} (目标: 20-50)")
        success1 = "✅" if 20 <= avg_phase1 <= 50 else "❌"
        print(f"  {success1} 阶段1效果: {'符合预期' if 20 <= avg_phase1 <= 50 else '不符合预期'}")
    
    if phase_data["phase2"]:
        avg_phase2 = np.mean(phase_data["phase2"])
        print(f"阶段2 (步骤{phase1_end+1}-{phase2_end}): 平均活跃灾难数 = {avg_phase2:.1f} (目标: 5-20)")
        success2 = "✅" if 5 <= avg_phase2 <= 20 else "❌"
        print(f"  {success2} 阶段2效果: {'符合预期' if 5 <= avg_phase2 <= 20 else '不符合预期'}")
    
    if phase_data["phase3"]:
        avg_phase3 = np.mean(phase_data["phase3"])
        print(f"阶段3 (步骤{phase2_end+1}-{max_steps}): 平均活跃灾难数 = {avg_phase3:.1f} (目标: 1-5)")
        success3 = "✅" if 1 <= avg_phase3 <= 5 else "❌"
        print(f"  {success3} 阶段3效果: {'符合预期' if 1 <= avg_phase3 <= 5 else '不符合预期'}")
    
    # 检查阶段间的趋势
    print("\n📈 阶段间趋势分析:")
    if phase_data["phase1"] and phase_data["phase2"]:
        print(f"阶段1→2: {np.mean(phase_data['phase1']):.1f} → {np.mean(phase_data['phase2']):.1f} ({'下降' if np.mean(phase_data['phase2']) < np.mean(phase_data['phase1']) else '上升'})")
    if phase_data["phase2"] and phase_data["phase3"]:
        print(f"阶段2→3: {np.mean(phase_data['phase2']):.1f} → {np.mean(phase_data['phase3']):.1f} ({'下降' if np.mean(phase_data['phase3']) < np.mean(phase_data['phase2']) else '上升'})")
    
    return phase_data

def analyze_latest_metadata():
    """分析最新的训练元数据"""
    print("\n🔍 分析最新训练元数据...")
    
    metadata_dir = Path("train_visualization_save/metadata")
    if not metadata_dir.exists():
        print("❌ 元数据目录不存在")
        return
    
    # 找到最新的元数据文件
    files = list(metadata_dir.glob("episode_*.json"))
    if not files:
        print("❌ 未找到元数据文件")
        return
    
    latest_file = sorted(files)[-1]
    print(f"📁 分析文件: {latest_file.name}")
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        print(f"📋 数据点总数: {len(data)}")
        
        if not data:
            print("❌ 元数据文件为空")
            return
        
        # 检查是否有新的字段
        first_entry = data[0]
        has_active_count = 'active_disaster_count' in first_entry
        has_seen_count = 'seen_disasters_count' in first_entry
        
        print(f"🔍 新字段检查:")
        print(f"  - active_disaster_count: {'✅' if has_active_count else '❌'}")
        print(f"  - seen_disasters_count: {'✅' if has_seen_count else '❌'}")
        
        if has_active_count:
            # 分析活跃灾难数的三阶段效果
            max_step = max(d['step'] for d in data)
            phase1_end = int(max_step * 2 / 3)
            phase2_end = int(max_step * 5 / 6)
            
            phase1_data = [d['active_disaster_count'] for d in data if d['step'] <= phase1_end]
            phase2_data = [d['active_disaster_count'] for d in data if phase1_end < d['step'] <= phase2_end]
            phase3_data = [d['active_disaster_count'] for d in data if d['step'] > phase2_end]
            
            print(f"\n📊 元数据中的三阶段分析 (最大步数: {max_step}):")
            
            if phase1_data:
                avg1 = np.mean(phase1_data)
                print(f"阶段1 (步骤1-{phase1_end}): 平均活跃灾难数 = {avg1:.1f} (目标: 20-50)")
            
            if phase2_data:
                avg2 = np.mean(phase2_data)
                print(f"阶段2 (步骤{phase1_end+1}-{phase2_end}): 平均活跃灾难数 = {avg2:.1f} (目标: 5-20)")
            
            if phase3_data:
                avg3 = np.mean(phase3_data)
                print(f"阶段3 (步骤{phase2_end+1}-{max_step}): 平均活跃灾难数 = {avg3:.1f} (目标: 1-5)")
        
        else:
            print("❌ 无法分析三阶段效果，缺少active_disaster_count字段")
            # 显示前几个数据点的结构
            print("\n📋 数据结构示例 (前3个数据点):")
            for i in range(min(3, len(data))):
                step_data = data[i]
                print(f"  步骤 {step_data.get('step', i+1)}: {list(step_data.keys())}")
    
    except Exception as e:
        print(f"❌ 读取元数据文件时出错: {e}")

if __name__ == "__main__":
    # 测试三阶段策略
    test_disaster_phases()
    
    # 分析最新元数据
    analyze_latest_metadata() 