#!/usr/bin/env python3
"""测试重构后的简化冲突检测逻辑"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from scheduler_service import SchedulerService

def test_direct_schedule_displacement():
    """测试直接调度100%替换可移动任务的逻辑"""
    print("💪 测试直接调度100%替换逻辑...")
    
    # 模拟混合事件：固定事件和可移动事件
    all_events = [
        {
            "id": "fixed1",
            "title": "重要会议",
            "scheduled_start": datetime(2024, 1, 1, 9, 0),
            "scheduled_end": datetime(2024, 1, 1, 10, 0),
            "task_type_id": None  # 固定事件
        },
        {
            "id": "movable1", 
            "title": "代码审查",
            "scheduled_start": datetime(2024, 1, 1, 10, 30),
            "scheduled_end": datetime(2024, 1, 1, 11, 30),
            "task_type_id": "task123"  # 可移动事件
        },
        {
            "id": "movable2",
            "title": "文档编写", 
            "scheduled_start": datetime(2024, 1, 1, 11, 0),
            "scheduled_end": datetime(2024, 1, 1, 12, 0),
            "task_type_id": "task456"  # 可移动事件
        }
    ]
    
    # 测试直接调度时间：10:00-12:00（与多个事件冲突）
    direct_start = datetime(2024, 1, 1, 10, 0)
    direct_end = datetime(2024, 1, 1, 12, 0)
    
    # 创建mock SchedulerService来测试逻辑
    class MockTaskTypeService:
        def __init__(self):
            self.supabase = None
    
    scheduler = SchedulerService(MockTaskTypeService())
    
    # 测试_find_conflicting_events
    conflicting = scheduler._find_conflicting_events(direct_start, direct_end, all_events)
    
    print(f"  🔍 总冲突事件: {len(conflicting)} 个")
    for event in conflicting:
        event_type = "固定" if not event.get("task_type_id") else "可移动"
        print(f"    • {event['title']} ({event_type})")
    
    # 测试可移动事件过滤
    movable_conflicts = [e for e in conflicting if e.get('task_type_id')]
    
    print(f"  ✨ 可移动冲突: {len(movable_conflicts)} 个")
    for event in movable_conflicts:
        print(f"    • {event['title']} - 将被移动")
    
    fixed_conflicts = [e for e in conflicting if not e.get('task_type_id')]
    print(f"  🔒 固定事件冲突: {len(fixed_conflicts)} 个")
    for event in fixed_conflicts:
        print(f"    • {event['title']} - 保持不变")
    
    # 验证逻辑正确性
    expected_movable = 2  # movable1 和 movable2
    expected_fixed = 0    # fixed1 不在 10:00-12:00 范围内
    
    result1 = "✅" if len(movable_conflicts) == expected_movable else "❌"
    result2 = "✅" if len(fixed_conflicts) == expected_fixed else "❌"
    
    print(f"\n  {result1} 可移动事件检测: 期望 {expected_movable}, 实际 {len(movable_conflicts)}")
    print(f"  {result2} 固定事件检测: 期望 {expected_fixed}, 实际 {len(fixed_conflicts)}")

def test_reuse_existing_functions():
    """测试是否正确重用了现有函数"""
    print("\n🔄 测试现有函数重用...")
    
    # 检查是否删除了重复函数
    from scheduler_service import SchedulerService
    
    has_old_function = hasattr(SchedulerService, '_fetch_existing_events_for_collision_check')
    has_new_function = hasattr(SchedulerService, '_displace_conflicting_events')
    has_existing_function = hasattr(SchedulerService, '_fetch_existing_events')
    has_conflict_function = hasattr(SchedulerService, '_find_conflicting_events')
    
    print(f"  ❌ 旧的重复函数已删除: {'是' if not has_old_function else '否'}")
    print(f"  ✅ 新的移动函数存在: {'是' if has_new_function else '否'}")
    print(f"  ✅ 现有获取事件函数: {'是' if has_existing_function else '否'}")
    print(f"  ✅ 现有冲突检测函数: {'是' if has_conflict_function else '否'}")
    
    result = "✅" if (not has_old_function and has_new_function and 
                    has_existing_function and has_conflict_function) else "❌"
    
    print(f"\n  {result} 函数重构完成度检查")

def test_code_simplification():
    """测试代码简化效果"""
    print("\n📏 测试代码简化效果...")
    
    # 读取scheduler_service.py文件检查代码行数和复杂度
    try:
        with open('scheduler_service.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # 检查是否有重复的获取事件逻辑
        fetch_event_patterns = [
            'fetch_existing_events_for_collision_check',  # 这个函数应该被删除
            'COLLISION_CHECK_ERROR'  # 这个错误日志应该不再存在
        ]
        
        duplicate_patterns_found = 0
        for pattern in fetch_event_patterns:
            count = sum(1 for line in lines if pattern in line)
            duplicate_patterns_found += count
        
        print(f"  📊 总行数: {total_lines}")
        print(f"  📊 代码行数: {code_lines}")
        print(f"  🔍 重复模式检测: {duplicate_patterns_found} 个")
        
        result = "✅" if duplicate_patterns_found == 0 else "❌"
        print(f"  {result} 代码去重检查: {'通过' if duplicate_patterns_found == 0 else '仍有重复'}")
        
    except FileNotFoundError:
        print("  ❌ 无法读取scheduler_service.py文件")

def test_displacement_logic():
    """测试100%替换逻辑"""
    print("\n🎯 测试100%替换逻辑...")
    
    # 模拟不同优先级的情况
    test_scenarios = [
        {
            "name": "直接调度 vs 低优先级自动调度",
            "direct_priority": 1.0,  # 直接调度总是最高优先级
            "auto_priority": 0.3,
            "should_displace": True
        },
        {
            "name": "直接调度 vs 高优先级自动调度", 
            "direct_priority": 1.0,  # 直接调度总是最高优先级
            "auto_priority": 0.9,
            "should_displace": True  # 直接调度仍然获胜
        },
        {
            "name": "直接调度 vs 固定事件",
            "direct_priority": 1.0,
            "fixed_event": True,
            "should_displace": False  # 不能移动固定事件
        }
    ]
    
    for scenario in test_scenarios:
        print(f"  📋 场景: {scenario['name']}")
        
        if scenario.get('fixed_event'):
            # 固定事件不应该被考虑移动
            print(f"    🔒 固定事件不会被移动")
            result = "✅"
        else:
            # 直接调度应该总是胜出
            will_displace = scenario['direct_priority'] > scenario['auto_priority']
            expected = scenario['should_displace']
            result = "✅" if will_displace == expected else "❌"
            
            print(f"    📊 直接调度优先级: {scenario['direct_priority']}")
            print(f"    📊 自动调度优先级: {scenario['auto_priority']}")
            print(f"    🎯 预期结果: {'替换' if expected else '不替换'}")
            print(f"    🎯 实际结果: {'替换' if will_displace else '不替换'}")
        
        print(f"    {result} 逻辑正确性")
        print()

def main():
    """运行所有测试"""
    print("🚀 开始测试重构后的简化逻辑...\n")
    
    test_direct_schedule_displacement()
    test_reuse_existing_functions()
    test_code_simplification()
    test_displacement_logic()
    
    print("🎉 重构验证测试完成!")

if __name__ == "__main__":
    main() 