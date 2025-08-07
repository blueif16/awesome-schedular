#!/usr/bin/env python3
"""简化的冲突检测测试脚本"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from scheduler_service import SchedulerService

def test_auto_schedule_detection():
    """测试auto_schedule参数检测"""
    print("🧪 测试auto_schedule检测逻辑...")
    
    # 模拟不同的输入情况
    test_cases = [
        {"start": None, "end": None, "expected": True},  # 自动调度
        {"start": "2024-01-01T10:00:00Z", "end": "2024-01-01T11:00:00Z", "expected": False},  # 直接调度
        {"start": "2024-01-01T10:00:00Z", "end": None, "expected": True},  # 部分时间，视为自动调度
        {"start": None, "end": "2024-01-01T11:00:00Z", "expected": True},  # 部分时间，视为自动调度
    ]
    
    for i, case in enumerate(test_cases, 1):
        start = case["start"]
        end = case["end"]
        expected = case["expected"]
        
        # 模拟scheduler_service.py中的逻辑
        is_auto_schedule = not (start and end)
        
        result = "✅" if is_auto_schedule == expected else "❌"
        schedule_type = "自动调度" if is_auto_schedule else "直接调度"
        expected_type = "自动调度" if expected else "直接调度"
        
        print(f"  测试用例 {i}: {result}")
        print(f"    输入: start={start}, end={end}")
        print(f"    检测结果: {schedule_type}")
        print(f"    期望结果: {expected_type}")
        print()

def test_collision_detection_logic():
    """测试冲突检测逻辑"""
    print("🔍 测试冲突检测逻辑...")
    
    # 模拟现有事件
    existing_events = [
        {
            "id": "event1",
            "title": "晨会",
            "scheduled_start": datetime(2024, 1, 1, 9, 0),
            "scheduled_end": datetime(2024, 1, 1, 9, 30)
        },
        {
            "id": "event2", 
            "title": "代码审查",
            "scheduled_start": datetime(2024, 1, 1, 11, 0),
            "scheduled_end": datetime(2024, 1, 1, 12, 0)
        }
    ]
    
    # 测试不同的新事件时间
    test_schedules = [
        {
            "name": "无冲突事件",
            "start": datetime(2024, 1, 1, 10, 0),
            "end": datetime(2024, 1, 1, 10, 30),
            "expected_conflicts": 0
        },
        {
            "name": "与晨会冲突",
            "start": datetime(2024, 1, 1, 9, 15),
            "end": datetime(2024, 1, 1, 10, 0),
            "expected_conflicts": 1
        },
        {
            "name": "跨越多个事件",
            "start": datetime(2024, 1, 1, 8, 30),
            "end": datetime(2024, 1, 1, 12, 30),
            "expected_conflicts": 2
        }
    ]
    
    for test in test_schedules:
        # 模拟_find_conflicting_events逻辑
        conflicting = []
        for event in existing_events:
            if (test["start"] < event['scheduled_end'] and 
                test["end"] > event['scheduled_start']):
                conflicting.append(event)
        
        result = "✅" if len(conflicting) == test["expected_conflicts"] else "❌"
        
        print(f"  {result} {test['name']}")
        print(f"    新事件时间: {test['start'].strftime('%H:%M')} - {test['end'].strftime('%H:%M')}")
        print(f"    检测到冲突: {len(conflicting)} 个")
        print(f"    期望冲突: {test['expected_conflicts']} 个")
        if conflicting:
            for conflict in conflicting:
                print(f"      冲突事件: {conflict['title']}")
        print()

def test_priority_calculation():
    """测试优先级计算逻辑"""
    print("📊 测试优先级计算逻辑...")
    
    # 创建一个最小的mock对象来测试计算逻辑
    class MockTaskTypeService:
        def __init__(self):
            self.supabase = None
    
    # 创建SchedulerService实例来测试_calculate_priority_score方法
    scheduler = SchedulerService(MockTaskTypeService())
    
    test_cases = [
        {
            "importance": 0.5,
            "deadline": None,
            "expected_range": (0.5, 0.5),
            "description": "无截止日期"
        },
        {
            "importance": 0.7,
            "deadline": datetime.now() + timedelta(hours=12),  # 12小时后
            "expected_range": (0.90, 0.92),  # 0.7 * 1.3 ≈ 0.91，允许小范围误差
            "description": "24小时内截止"
        },
        {
            "importance": 0.6,
            "deadline": datetime.now() + timedelta(days=2),  # 2天后
            "expected_range": (0.66, 0.66),  # 0.6 * 1.1 = 0.66
            "description": "3天内截止"
        },
        {
            "importance": 0.8,
            "deadline": datetime.now() - timedelta(hours=1),  # 已过期
            "expected_range": (1.0, 1.0),  # min(1.0, 0.8 * 1.5) = 1.0
            "description": "已过期"
        }
    ]
    
    for test in test_cases:
        priority = scheduler._calculate_priority_score(test["importance"], test["deadline"])
        expected_min, expected_max = test["expected_range"]
        
        result = "✅" if expected_min <= priority <= expected_max else "❌"
        
        print(f"  {result} {test['description']}")
        print(f"    重要性分数: {test['importance']}")
        print(f"    计算优先级: {priority:.3f}")
        print(f"    期望范围: {expected_min:.3f} - {expected_max:.3f}")
        print()

def main():
    """运行所有测试"""
    print("🚀 开始简化功能测试...\n")
    
    test_auto_schedule_detection()
    test_collision_detection_logic()
    test_priority_calculation()
    
    print("🎉 测试完成!")

if __name__ == "__main__":
    main() 