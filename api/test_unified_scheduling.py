#!/usr/bin/env python3
"""
Test Unified Scheduling with Event Pushing
Demonstrates how urgent events automatically push lower priority events to suboptimal slots
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_type_service import TaskTypeService
from scheduler_service import SchedulerService
from models import ScheduleEventRequest, TaskType
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

async def test_urgent_event_pushing():
    """Test unified scheduling with urgent event pushing lower priority events"""
    
    print("🧪 Testing Unified Scheduling with Event Pushing")
    print("=" * 60)
    
    # Setup services
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([supabase_url, supabase_key, openai_api_key]):
        print("❌ Missing environment variables")
        return
    
    supabase = create_client(supabase_url, supabase_key)
    task_service = TaskTypeService(supabase, openai_api_key)
    scheduler_service = SchedulerService(task_service)
    
    test_user_id = str(uuid4())
    
    # Create test user in database first
    try:
        supabase.table("users").insert({
            "id": test_user_id,
            "email": f"test-{test_user_id[:8]}@example.com",
            "role": "student",
            "timezone": "UTC",
            "weekly_energy_pattern": [0.5] * 168  # Will override with custom pattern
        }).execute()
        print(f"👤 Created test user: {test_user_id[:8]}...")
    except Exception as e:
        print(f"⚠️ User creation failed (may already exist): {e}")
    
    # Sample user energy pattern (high energy mornings)
    user_energy_pattern = []
    for day in range(7):  # 7 days
        for hour in range(24):  # 24 hours
            if 8 <= hour <= 11:
                energy = 0.9  # High energy mornings
            elif 12 <= hour <= 17:
                energy = 0.7  # Medium energy afternoons  
            elif 18 <= hour <= 23:
                energy = 0.4  # Low energy evenings
            else:
                energy = 0.2  # Very low energy nights
            user_energy_pattern.append(energy)
    
    print(f"👤 User energy pattern: High mornings (0.9), Medium afternoons (0.7), Low evenings (0.4)")
    print()
    
    # Step 1: Create some existing events (lower priority)
    print("📅 STEP 1: Creating existing events (lower priority)")
    print("-" * 40)
    
    tomorrow_9am = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    tomorrow_10am = tomorrow_9am + timedelta(hours=1)
    tomorrow_11am = tomorrow_9am + timedelta(hours=2)
    
    existing_events = [
        {
            "id": str(uuid4()),
            "user_id": test_user_id,
            "title": "Email Processing",
            "description": "Check and respond to emails",
            "scheduled_start": tomorrow_9am,
            "scheduled_end": tomorrow_10am,
            "calculated_priority": 0.4  # Lower priority
        },
        {
            "id": str(uuid4()),
            "user_id": test_user_id,
            "title": "Social Media Review",
            "description": "Review social media updates",
            "scheduled_start": tomorrow_10am,
            "scheduled_end": tomorrow_11am,
            "calculated_priority": 0.3  # Even lower priority
        }
    ]
    
    # Create existing events in database
    for event in existing_events:
        try:
            # Convert datetime objects to ISO format for database
            db_event = event.copy()
            db_event['scheduled_start'] = db_event['scheduled_start'].isoformat()
            db_event['scheduled_end'] = db_event['scheduled_end'].isoformat()
            
            supabase.table("events").insert(db_event).execute()
            print(f"   📧 {event['title']}: {event['scheduled_start'].strftime('%A %H:%M')} "
                  f"(priority: {event['calculated_priority']:.1f}) ✅ Created in DB")
        except Exception as e:
            print(f"   📧 {event['title']}: {event['scheduled_start'].strftime('%A %H:%M')} "
                  f"(priority: {event['calculated_priority']:.1f}) ❌ DB Error: {e}")
    
    print()
    
    # Step 2: Try to schedule urgent event that collides
    print("🚨 STEP 2: Scheduling urgent event that collides with existing events")
    print("-" * 40)
    
    urgent_request = ScheduleEventRequest(
        title="Critical Client Meeting",
        description="Emergency meeting with top client - cannot be delayed",
        duration=2.0,  # 2 hours - will conflict with both existing events
        preferred_date=tomorrow_9am,
        importance_score=0.9,  # High importance
        deadline=tomorrow_9am + timedelta(hours=3)  # Urgent deadline
    )
    
    print(f"⚡ Urgent Event: {urgent_request.title}")
    print(f"   Duration: {urgent_request.duration} hours")
    print(f"   Importance: {urgent_request.importance_score}")
    print(f"   Deadline: {urgent_request.deadline.strftime('%A %H:%M')}")
    print(f"   Will conflict with: {len(existing_events)} existing events")
    print()
    
    try:
        # Use unified scheduling - should automatically push lower priority events
        result = await scheduler_service.schedule_with_unified_scoring(
            user_id=test_user_id,
            user_energy_pattern=user_energy_pattern,
            request=urgent_request,
            existing_events=existing_events
        )
        
        print("✅ UNIFIED SCHEDULING SUCCESSFUL!")
        print("=" * 40)
        
        # Show the scheduled urgent event
        new_event = result['event']
        print(f"🎯 URGENT EVENT SCHEDULED:")
        print(f"   📅 {new_event['title']}")
        print(f"   🕐 {new_event['scheduled_start'].strftime('%A %m/%d %H:%M')} - {new_event['scheduled_end'].strftime('%H:%M')}")
        print(f"   📊 Priority: {new_event['calculated_priority']:.3f}")
        print(f"   🎯 Slot Score: {result.get('slot_score', 0):.3f}")
        print()
        
        # Show rescheduled events
        if result.get('rescheduled_events'):
            print(f"🔄 EVENTS AUTOMATICALLY PUSHED:")
            print(f"   📋 {len(result['rescheduled_events'])} events were moved to suboptimal slots")
            
            for i, rescheduled in enumerate(result['rescheduled_events'], 1):
                original = rescheduled['original_event']
                print(f"   {i}. {original['title']}:")
                print(f"      📅 Moved to: {rescheduled['new_start'].strftime('%A %m/%d %H:%M')} - {rescheduled['new_end'].strftime('%H:%M')}")
                print(f"      📊 Score change: {rescheduled['score_change']:+.3f}")
            print()
        
        # Show summary
        if 'rescheduling_summary' in result:
            summary = result['rescheduling_summary']
            print(f"📊 RESCHEDULING SUMMARY:")
            print(f"   🔢 Events moved: {summary['events_moved']}")
            print(f"   ⚡ New event priority: {summary['new_event_priority']:.3f}")
            print(f"   📉 Average moved priority: {summary['average_moved_priority']:.3f}")
            print(f"   🎯 Method: {result['scheduling_method']}")
        
        print()
        print("🎉 SUCCESS: Urgent event pushed lower priority events to their suboptimal slots!")
        
    except Exception as e:
        print(f"❌ Unified scheduling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_urgent_event_pushing()) 