#!/usr/bin/env python3
"""
测试增强后的智能体状态表示功能
验证智能体能获取的所有信息，包括灾难点详情和其他智能体信息
"""

import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.environment import Environment
from src.rl.marl_rescue import MARLController
from src.core import config

def test_enhanced_state_representation():
    """测试增强后的状态表示功能"""
    print("🔍 测试增强后的智能体状态表示功能...")
    
    # 创建环境
    env = Environment(grid_size=20, num_rescuers=5, training_mode=True)
    
    # 创建MARL控制器
    controller = MARLController(
        env_or_grid_size=env.GRID_SIZE,
        num_rescuers=len(env.rescuers)
    )
    
    # 等待几步让灾难产生并让智能体选择目标
    for step in range(5):
        env.update_disasters_silent(current_time_step=step)
        
        # 让智能体选择动作
        actions = controller.select_actions(env.rescuers, env.disasters, training=False)
    
    print(f"\n📊 环境状态信息:")
    print(f"• 网格大小: {env.GRID_SIZE}x{env.GRID_SIZE}")
    print(f"• 救援人员数量: {len(env.rescuers)}")
    print(f"• 当前灾难点数量: {len(env.disasters)}")
    
    # 分析第一个智能体的状态
    rescuer_idx = 0
    state_tensor = controller.build_state(rescuer_idx, env.rescuers, env.disasters)
    state_np = state_tensor.cpu().numpy().flatten()
    
    print(f"\n🧠 智能体 {rescuer_idx} 的状态信息分析:")
    print(f"• 状态向量总维度: {len(state_np)}")
    print(f"• 预期维度: {controller.state_dim}")
    
    # 解析自身状态信息
    self_state = state_np[:4]
    print(f"\n👤 自身状态信息:")
    print(f"• 位置 X (标准化): {self_state[0]:.3f} (实际: {env.rescuers[rescuer_idx]['position'][0]})")
    print(f"• 位置 Y (标准化): {self_state[1]:.3f} (实际: {env.rescuers[rescuer_idx]['position'][1]})")
    print(f"• 是否有目标: {self_state[2]:.1f} ({'是' if self_state[2] > 0.5 else '否'})")
    print(f"• 移动速度 (标准化): {self_state[3]:.3f}")
    
    # 解析网格状态信息
    grid_state = state_np[4:].reshape((env.GRID_SIZE, env.GRID_SIZE, 7))
    
    print(f"\n🗺️ 网格状态信息 (7个通道):")
    print(f"• 通道0: 灾情存在标志")
    print(f"• 通道1: 灾情等级 (衰减等级)")
    print(f"• 通道2: 所需救援次数")
    print(f"• 通道3: 已分配智能体数量")
    print(f"• 通道4: 剩余所需救援次数 (与通道2相同)")
    print(f"• 通道5: 其他智能体位置密度")
    print(f"• 通道6: 其他智能体目标位置密度")
    
    # 找到有灾难的位置
    disaster_positions = []
    for x in range(env.GRID_SIZE):
        for y in range(env.GRID_SIZE):
            if grid_state[x, y, 0] > 0:  # 通道0: 灾情存在
                disaster_positions.append((x, y))
    
    print(f"\n🚨 检测到的灾难点信息 ({len(disaster_positions)} 个):")
    for i, (x, y) in enumerate(disaster_positions[:5]):  # 只显示前5个
        disaster_exists = grid_state[x, y, 0]
        disaster_level = grid_state[x, y, 1] * config.CRITICAL_DISASTER_THRESHOLD
        rescue_needed = grid_state[x, y, 2] * config.MAX_RESCUE_CAPACITY
        assigned_agents = grid_state[x, y, 3] * len(env.rescuers)
        
        print(f"  位置({x}, {y}): 等级={disaster_level:.1f}, 需救援={rescue_needed:.1f}, 已分配={assigned_agents:.1f}个智能体")
        
        # 验证与实际环境数据的一致性
        if (x, y) in env.disasters:
            actual_disaster = env.disasters[(x, y)]
            print(f"    验证: 实际等级={actual_disaster['level']:.1f}, 实际需救援={actual_disaster['rescue_needed']:.1f}")
    
    # 分析其他智能体位置信息
    print(f"\n👥 其他智能体信息:")
    other_agent_positions = []
    other_agent_targets = []
    
    for x in range(env.GRID_SIZE):
        for y in range(env.GRID_SIZE):
            if grid_state[x, y, 5] > 0:  # 通道5: 其他智能体位置
                density = grid_state[x, y, 5]
                estimated_count = density * (len(env.rescuers) - 1)
                other_agent_positions.append((x, y, estimated_count))
            
            if grid_state[x, y, 6] > 0:  # 通道6: 其他智能体目标
                density = grid_state[x, y, 6]
                estimated_count = density * (len(env.rescuers) - 1)
                other_agent_targets.append((x, y, estimated_count))
    
    print(f"• 其他智能体位置 ({len(other_agent_positions)} 个):")
    for x, y, count in other_agent_positions:
        print(f"  位置({x}, {y}): 约{count:.1f}个智能体")
    
    print(f"• 其他智能体目标 ({len(other_agent_targets)} 个):")
    for x, y, count in other_agent_targets:
        print(f"  目标({x}, {y}): 约{count:.1f}个智能体前往")
    
    # 验证其他智能体信息的准确性
    print(f"\n✅ 验证其他智能体信息:")
    for i, rescuer in enumerate(env.rescuers):
        if i != rescuer_idx:
            pos = rescuer['position']
            target = rescuer.get('target', None)
            print(f"• 智能体{i}: 位置{pos}" + (f", 目标{target}" if target else ", 无目标"))
    
    return True

def analyze_state_information_detail():
    """详细分析智能体能获取的信息内容"""
    print("\n" + "="*60)
    print("📋 智能体状态信息详细分析")
    print("="*60)
    
    print("✅ 智能体可以获取的完整信息:")
    
    print("\n1️⃣ 关于每个灾难点:")
    print("   • 是否存在灾难 (0或1)")
    print("   • 灾难等级/衰减程度 (0-1标准化)")
    print("   • 所需救援次数 (0-1标准化)")
    print("   • 有多少个智能体将此处设为目标 (0-1标准化)")
    print("   • 剩余需要的救援次数 (与所需救援次数相同)")
    
    print("\n2️⃣ 关于其他智能体:")
    print("   • 其他智能体的当前位置分布")
    print("   • 其他智能体的目标位置分布")
    print("   • 通过位置密度可推断具体坐标")
    
    print("\n3️⃣ 关于自身:")
    print("   • 自己的当前位置坐标")
    print("   • 是否已分配目标任务")
    print("   • 自己的移动速度")
    
    print("\n✅ 回答您的问题:")
    print("❓ 能得到灾难点的所需救援次数吗? ✅ 能！通过通道2和通道4")
    print("❓ 能得到剩余衰减等级吗? ✅ 能！通过通道1")
    print("❓ 能得知其他智能体正在前往的坐标吗? ✅ 能！通过通道3和通道6")
    
    print("\n💡 增强功能:")
    print("   • 新增通道5: 可直接看到其他智能体的位置")
    print("   • 新增通道6: 可直接看到其他智能体的目标分布")
    print("   • 原有通道3: 可知道每个灾难点有多少智能体前往")

if __name__ == "__main__":
    try:
        # 测试增强后的状态表示
        test_enhanced_state_representation()
        
        # 分析详细信息
        analyze_state_information_detail()
        
        print(f"\n🎉 测试完成！智能体现在可以获取更丰富的环境信息。")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc() 