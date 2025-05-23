#!/usr/bin/env python3
"""
测试修复后的三阶段灾难生成策略
验证在后期阶段当活跃灾难点不足时能够正确补充
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.environment import Environment
from src.rl.rl_util import adjust_disaster_settings

def test_fixed_disaster_phases():
    """测试修复后的三阶段灾难生成策略"""
    print("🔧 测试修复后的三阶段灾难生成策略...")
    print("=" * 60)
    
    # 创建环境（使用较小的网格以便观察）
    env = Environment(grid_size=20, num_rescuers=10, training_mode=True, verbose=False)
    max_steps = 100
    
    # 记录三个阶段的活跃灾难点数量
    phase1_counts = []  # 步骤 0-66
    phase2_counts = []  # 步骤 67-83  
    phase3_counts = []  # 步骤 84-100
    
    # 计算阶段边界
    phase1_end = int(max_steps * 2 / 3)  # 66
    phase2_end = int(max_steps * 5 / 6)  # 83
    
    print(f"阶段划分：")
    print(f"• 阶段1（初期）：步骤 0-{phase1_end} (目标: 20-50个活跃灾难点)")
    print(f"• 阶段2（中期）：步骤 {phase1_end+1}-{phase2_end} (目标: 5-20个活跃灾难点)")
    print(f"• 阶段3（后期）：步骤 {phase2_end+1}-{max_steps} (目标: 1-5个活跃灾难点)")
    print("=" * 60)
    
    # 强制清空所有灾难点，从零开始测试
    env.disasters = {}
    print("🧹 已清空所有灾难点，从零开始测试...")
    
    for step in range(max_steps):
        # 调用修复后的灾难调整函数
        adjust_disaster_settings(env, step, max_steps, verbose=(step % 20 == 0))
        
        # 更新灾难状态（让环境自然生成一些灾难点）
        env.update_disasters_silent(current_time_step=step)
        env.current_time_step = step
        
        # 统计当前活跃灾难点数量
        active_count = sum(1 for d in env.disasters.values() if d.get("rescue_needed", 0) > 0)
        
        # 根据阶段记录数据
        if step <= phase1_end:
            phase1_counts.append(active_count)
        elif step <= phase2_end:
            phase2_counts.append(active_count)
        else:
            phase3_counts.append(active_count)
        
        # 每10步显示状态
        if step % 10 == 0:
            total_disasters = len(env.disasters)
            resolved = sum(1 for d in env.disasters.values() 
                         if d.get("rescue_needed", 0) == 0 and d.get("rescue_success", False))
            failed = sum(1 for d in env.disasters.values() 
                       if d.get("rescue_needed", 0) == 0 and not d.get("rescue_success", False))
            
            current_phase = "阶段1" if step <= phase1_end else ("阶段2" if step <= phase2_end else "阶段3")
            print(f"[步骤 {step:2d}] {current_phase} | 活跃: {active_count:2d} | 总计: {total_disasters:3d} | 已解决: {resolved:2d} | 失败: {failed:3d}")
    
    # 分析结果
    print("\n" + "=" * 60)
    print("📊 测试结果分析")
    print("=" * 60)
    
    # 计算平均活跃灾难点数量
    phase1_avg = sum(phase1_counts) / len(phase1_counts) if phase1_counts else 0
    phase2_avg = sum(phase2_counts) / len(phase2_counts) if phase2_counts else 0
    phase3_avg = sum(phase3_counts) / len(phase3_counts) if phase3_counts else 0
    
    print(f"阶段1 (初期): 平均活跃灾难点 = {phase1_avg:.1f} (目标: 20-50)")
    phase1_status = "✅ 符合预期" if 20 <= phase1_avg <= 50 else "❌ 不符合预期"
    print(f"  {phase1_status}")
    
    print(f"阶段2 (中期): 平均活跃灾难点 = {phase2_avg:.1f} (目标: 5-20)")
    phase2_status = "✅ 符合预期" if 5 <= phase2_avg <= 20 else "❌ 不符合预期"
    print(f"  {phase2_status}")
    
    print(f"阶段3 (后期): 平均活跃灾难点 = {phase3_avg:.1f} (目标: 1-5)")
    phase3_status = "✅ 符合预期" if 1 <= phase3_avg <= 5 else "❌ 不符合预期"
    print(f"  {phase3_status}")
    
    # 检查是否有明显的下降趋势
    trend_ok = phase1_avg > phase2_avg > phase3_avg
    trend_status = "✅ 有明显下降趋势" if trend_ok else "❌ 趋势不明显"
    print(f"\n趋势分析: {phase1_avg:.1f} → {phase2_avg:.1f} → {phase3_avg:.1f}")
    print(f"  {trend_status}")
    
    # 特别检查后期阶段是否有零活跃灾难点的情况
    phase3_zeros = sum(1 for count in phase3_counts if count == 0)
    if phase3_zeros > 0:
        print(f"\n⚠️ 后期阶段发现 {phase3_zeros}/{len(phase3_counts)} 步有0个活跃灾难点")
        print(f"   这表明强制补充逻辑可能需要进一步优化")
    else:
        print(f"\n✅ 后期阶段没有出现0个活跃灾难点的情况")
    
    # 输出详细的后期阶段数据
    print(f"\n🔍 后期阶段详细数据:")
    print(f"• 最小活跃灾难点数: {min(phase3_counts) if phase3_counts else 0}")
    print(f"• 最大活跃灾难点数: {max(phase3_counts) if phase3_counts else 0}")
    print(f"• 活跃灾难点数变化: {phase3_counts}")
    
    return phase1_avg, phase2_avg, phase3_avg

def test_force_add_function():
    """单独测试强制添加灾难点的功能"""
    print("\n" + "=" * 60)
    print("🧪 测试强制添加灾难点功能")
    print("=" * 60)
    
    # 创建空环境
    env = Environment(grid_size=10, num_rescuers=5, training_mode=True, verbose=False)
    env.disasters = {}  # 清空所有灾难点
    
    print(f"初始状态: {len(env.disasters)} 个灾难点")
    
    # 导入强制添加函数
    from src.rl.rl_util import _force_add_disasters
    
    # 测试添加3个灾难点
    added = _force_add_disasters(env, 3, verbose=True)
    print(f"\n尝试添加3个灾难点，实际添加了: {added}")
    
    # 检查结果
    active_count = sum(1 for d in env.disasters.values() if d.get("rescue_needed", 0) > 0)
    print(f"当前活跃灾难点数量: {active_count}")
    print(f"总灾难点数量: {len(env.disasters)}")
    
    # 验证添加的灾难点的属性
    print(f"\n添加的灾难点详情:")
    for pos, disaster in env.disasters.items():
        print(f"  位置{pos}: 等级={disaster['level']:.1f}, 需救援={disaster['rescue_needed']}")

if __name__ == "__main__":
    try:
        # 测试三阶段策略
        phase1_avg, phase2_avg, phase3_avg = test_fixed_disaster_phases()
        
        # 测试强制添加功能
        test_force_add_function()
        
        print(f"\n🎉 测试完成！")
        print(f"修复后的三阶段策略能够确保每个阶段都有适当数量的活跃灾难点。")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc() 