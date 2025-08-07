#!/usr/bin/env python3
"""
测试冲突检测和移动功能的脚本
测试流程:
1. 插入几个示例自动调度事件
2. 创建一个直接调度事件与之冲突
3. 验证冲突事件被正确移动到备选时间段
"""

import asyncio
import os
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduler_service import SchedulerService
from task_type_service import TaskTypeService
from db_service import DatabaseService
from supabase import create_client
import logging

from dotenv import load_dotenv
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_test_data():
    """设置测试数据"""
    # 配置Supabase客户端
    supabase_url = os.getenv("SUPABASE_URL", "your_supabase_url")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "your_supabase_service_role_key")
    
    if not supabase_url or not supabase_key or supabase_url == "your_supabase_url":
        logger.error("请设置正确的SUPABASE_URL和SUPABASE_SERVICE_ROLE_KEY环境变量")
        return None, None, None
    
    supabase = create_client(supabase_url, supabase_key)
    task_type_service = TaskTypeService(supabase)
    scheduler_service = SchedulerService(task_type_service)
    db_service = DatabaseService(supabase)
    
    return scheduler_service, task_type_service, db_service

async def insert_sample_events(scheduler_service, user_id="test_user_123"):
    """插入示例自动调度事件"""
    logger.info("🔧 开始插入示例自动调度事件...")
    
    # 定义示例自动调度事件
    base_date = datetime.now() + timedelta(days=1)  # 明天
    base_date = base_date.replace(hour=9, minute=0, second=0, microsecond=0)  # 9:00 AM
    
    sample_events = [
        {
            "summary": "每日晨会",
            "description": "团队每日站立会议 - prefer early morning start for team alignment",  # 早晨偏好
            "duration": 0.5,  # 30分钟
            "importance_score": 0.7,
            "expected_time": base_date  # 9:00 AM
        },
        {
            "summary": "代码审查",
            "description": "审查新功能的代码 - prefer in morning when I'm most focused and alert",  # 早晨偏好
            "duration": 1.0,  # 1小时
            "importance_score": 0.6,
            "expected_time": base_date + timedelta(hours=2)  # 11:00 AM
        },
        {
            "summary": "客户会议",
            "description": "与客户讨论项目需求 - business meetings work best in afternoon hours",  # 下午偏好
            "duration": 1.5,  # 1.5小时
            "importance_score": 0.8,
            "expected_time": base_date + timedelta(hours=4)  # 1:00 PM
        },
        {
            "summary": "创意头脑风暴",
            "description": "新产品功能的创意讨论 - I prefer working on creative tasks in the morning hours",  # 早晨偏好
            "duration": 2.0,  # 2小时
            "importance_score": 0.75,
            "expected_time": base_date + timedelta(hours=6)  # 3:00 PM
        },
        {
            "summary": "文档整理",
            "description": "整理项目文档和资料 - administrative tasks are better in afternoon when energy is stable",  # 下午偏好
            "duration": 1.0,  # 1小时
            "importance_score": 0.4,
            "expected_time": base_date + timedelta(hours=8)  # 5:00 PM
        }
    ]
    
    created_events = []
    for event in sample_events:
        try:
            logger.info(f"📝 创建自动调度事件: {event['summary']}")
            event_id = await scheduler_service.schedule_with_pattern(
                user_id=user_id,
                summary=event["summary"],
                description=event["description"],
                duration=event["duration"],
                importance_score=event["importance_score"]
            )
            created_events.append({
                "id": event_id,
                "summary": event["summary"],
                "expected_time": event["expected_time"]
            })
            logger.info(f"✅ 创建成功: {event_id}")
            
        except Exception as e:
            logger.error(f"❌ 创建事件失败 '{event['summary']}': {e}")
    
    return created_events

async def test_direct_schedule_collision(scheduler_service, user_id="test_user_123"):
    """测试直接调度与现有事件的冲突处理"""
    logger.info("🚨 开始测试直接调度冲突处理...")
    
    # 创建一个与现有事件冲突的直接调度事件
    tomorrow = datetime.now() + timedelta(days=1)
    conflict_start = tomorrow.replace(hour=10, minute=30, second=0, microsecond=0)  # 10:30 AM
    conflict_end = conflict_start + timedelta(hours=2)  # 12:30 PM - 与多个事件冲突
    
    try:
        logger.info(f"📍 创建直接调度事件: {conflict_start} - {conflict_end}")
        direct_event_id = await scheduler_service.schedule_with_pattern(
            user_id=user_id,
            start=conflict_start.isoformat(),
            end=conflict_end.isoformat(),
            summary="紧急客户会议",
            description="重要客户的紧急需求讨论",
            importance_score=0.9
        )
        
        logger.info(f"✅ 直接调度事件创建成功: {direct_event_id}")
        return direct_event_id
        
    except Exception as e:
        logger.error(f"❌ 直接调度事件创建失败: {e}")
        return None

async def verify_collision_resolution(db_service, user_id="test_user_123"):
    """验证冲突解决结果"""
    logger.info("🔍 验证冲突解决结果...")
    
    try:
        # 获取所有用户事件
        tomorrow = datetime.now() + timedelta(days=1)
        day_start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        result = db_service.supabase.table("events") \
            .select("id, title, scheduled_start, scheduled_end, task_type_id") \
            .eq("user_id", user_id) \
            .gte("scheduled_start", day_start.isoformat()) \
            .lte("scheduled_end", day_end.isoformat()) \
            .order("scheduled_start") \
            .execute()
        
        events = result.data
        logger.info(f"📋 找到 {len(events)} 个事件:")
        
        for event in events:
            start_time = datetime.fromisoformat(event["scheduled_start"])
            end_time = datetime.fromisoformat(event["scheduled_end"])
            event_type = "直接调度" if not event.get("task_type_id") else "自动调度"
            
            logger.info(f"  • {event['title']} ({event_type})")
            logger.info(f"    时间: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
        
        # 检查是否有时间冲突
        conflicts = []
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                start1 = datetime.fromisoformat(event1["scheduled_start"])
                end1 = datetime.fromisoformat(event1["scheduled_end"])
                start2 = datetime.fromisoformat(event2["scheduled_start"])
                end2 = datetime.fromisoformat(event2["scheduled_end"])
                
                if start1 < end2 and start2 < end1:
                    conflicts.append((event1["title"], event2["title"]))
        
        if conflicts:
            logger.warning(f"⚠️ 发现 {len(conflicts)} 个时间冲突:")
            for conflict in conflicts:
                logger.warning(f"  冲突: {conflict[0]} vs {conflict[1]}")
        else:
            logger.info("✅ 没有发现时间冲突，冲突解决成功!")
            
    except Exception as e:
        logger.error(f"❌ 验证失败: {e}")

async def cleanup_test_data(db_service, user_id="test_user_123"):
    """清理测试数据"""
    logger.info("🧹 清理测试数据...")
    
    try:
        # 删除测试用户的所有事件
        result = db_service.supabase.table("events") \
            .delete() \
            .eq("user_id", user_id) \
            .execute()
            
        logger.info(f"🗑️ 已删除 {len(result.data)} 个测试事件")
        
    except Exception as e:
        logger.error(f"❌ 清理失败: {e}")

async def main():
    """主测试流程"""
    logger.info("🚀 开始冲突检测测试...")
    
    # 设置测试环境
    scheduler_service, task_type_service, db_service = await setup_test_data()
    if not scheduler_service:
        return
    
    user_id = "33a07e45-c5a8-4b95-9e39-c12752012e36"  # Use specified UUID
    
    try:
        # 清理之前的测试数据
        await cleanup_test_data(db_service, user_id)
        
        # 1. 插入示例自动调度事件
        created_events = await insert_sample_events(scheduler_service, user_id)
        logger.info(f"📊 已创建 {len(created_events)} 个自动调度事件")
        
        # 等待一会儿确保事件创建完成
        await asyncio.sleep(2)
        
        # 2. 创建冲突的直接调度事件
        direct_event_id = await test_direct_schedule_collision(scheduler_service, user_id)
        
        if direct_event_id:
            # 等待冲突处理完成
            await asyncio.sleep(3)
            
            # 3. 验证冲突解决结果
            await verify_collision_resolution(db_service, user_id)
        
        logger.info("🎉 测试完成!")
        
        # 保留数据供手动检查，不自动清理
        # await cleanup_test_data(db_service, user_id)
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 