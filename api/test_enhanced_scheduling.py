#!/usr/bin/env python3
"""
Test Enhanced Behavioral Scheduling
Validates that the new behavioral pattern scheduling works correctly
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

async def test_enhanced_behavioral_scheduling():
    """Test the enhanced behavioral scheduling with sample data"""
    
    print("🧪 Testing Enhanced Behavioral Scheduling")
    print("=" * 50)
    
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
    
    # Test user ID (use existing demo user or create mock)
    test_user_id = str(uuid4())
    
    # Sample user energy pattern (168 elements: 7 days × 24 hours)
    # High energy mornings (8-11), medium afternoons (12-17), low evenings (18-23)
    user_energy_pattern = []
    for day in range(7):  # 7 days
        for hour in range(24):  # 24 hours
            if 8 <= hour <= 11:
                energy = 0.9  # High energy mornings
            elif 12 <= hour <= 17:
                energy = 0.7  # Medium energy afternoons  
            elif 18 <= hour <= 23:
                energy = 0.3  # Low energy evenings
            else:
                energy = 0.2  # Very low energy nights/early morning
            user_energy_pattern.append(energy)
    
    print(f"👤 User energy pattern: {len(user_energy_pattern)} elements")
    print(f"🌅 Morning energy (8-11): {user_energy_pattern[8]:.1f}")
    print(f"🌞 Afternoon energy (12-17): {user_energy_pattern[12]:.1f}")
    print(f"🌙 Evening energy (18-23): {user_energy_pattern[18]:.1f}")
    print()
    
    # Create sample task type with behavioral patterns
    print("📝 Creating sample task type with behavioral patterns...")
    
    # Sample task type: "Deep Work Coding Session"
    sample_task_type = TaskType(
        id=uuid4(),
        user_id=test_user_id,
        task_type="Deep Work Coding Session",
        description="Intensive coding session requiring high focus",
        
        # Weekly habit scores: prefers weekday mornings (9-11)
        weekly_habit_scores=[0.2] * 168,  # Start with low preference everywhere
        
        # 7x24 confidence matrix: high confidence for weekday mornings
        slot_confidence=[[0.1] * 24 for _ in range(7)],  # Start with low confidence
        
        completion_count=5,
        typical_duration=2.0,     # Typically 2 hours
        cognitive_load=0.9,       # Very high cognitive load
        recovery_hours=1.0,       # Needs 1 hour recovery
        importance_score=0.8,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Set high preference for weekday mornings (Mon-Fri, 9-11 AM)
    for day in range(1, 6):  # Monday=1 to Friday=5 (0=Sunday)
        for hour in range(9, 12):  # 9-11 AM
            weekly_index = day * 24 + hour
            sample_task_type.weekly_habit_scores[weekly_index] = 0.9
            sample_task_type.slot_confidence[day][hour] = 0.8
    
    print(f"✅ Task type: {sample_task_type.task_type}")
    print(f"🧠 Cognitive load: {sample_task_type.cognitive_load}")
    print(f"⏱️ Typical duration: {sample_task_type.typical_duration}h")
    print(f"🔄 Recovery needed: {sample_task_type.recovery_hours}h")
    print()
    
    # Test 1: Schedule during high-energy, high-preference time
    print("🧪 Test 1: High-cognitive task during optimal time")
    print("-" * 40)
    
    request = ScheduleEventRequest(
        title="Deep Work Coding Session",
        description="Focus coding session",
        duration=2.0,
        preferred_date=datetime.now() + timedelta(days=1)  # Tomorrow
    )
    
    try:
        # Mock the find_similar_task_type to return our sample
        class MockSimilarResult:
            def __init__(self, task_type, similarity):
                self.task_type = task_type
                self.similarity = similarity
        
        # Patch the task service method for testing
        original_method = task_service.find_similar_task_type
        async def mock_find_similar(user_id, query):
            return MockSimilarResult(sample_task_type, 0.9)
        
        task_service.find_similar_task_type = mock_find_similar
        
        result = await scheduler_service.schedule_with_behavioral_patterns(
            user_id=test_user_id,
            user_energy_pattern=user_energy_pattern,
            request=request,
            existing_events=[]
        )
        
        # Restore original method
        task_service.find_similar_task_type = original_method
        
        print(f"✅ Scheduling successful!")
        print(f"📅 Scheduled: {result['event']['scheduled_start'].strftime('%A %H:%M')}")
        print(f"📊 Priority score: {result['event']['calculated_priority']:.3f}")
        print(f"🎯 Method: {result['scheduling_method']}")
        
        if 'optimal_slot' in result:
            slot = result['optimal_slot']
            print(f"💡 Reasoning: {slot.get('reasoning', 'N/A')}")
            print(f"🎯 Score: {slot.get('score', 0):.3f}")
        
        print()
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        print()
    
    # Test 2: Compare different cognitive loads
    print("🧪 Test 2: Low-cognitive task (should prefer different times)")
    print("-" * 40)
    
    # Create low-cognitive task type
    low_cognitive_task = TaskType(
        id=uuid4(),
        user_id=test_user_id,
        task_type="Email Processing",
        description="Process and respond to emails",
        weekly_habit_scores=[0.5] * 168,  # Neutral preferences
        slot_confidence=[[0.3] * 24 for _ in range(7)],  # Medium confidence
        completion_count=20,
        typical_duration=0.5,     # 30 minutes
        cognitive_load=0.2,       # Low cognitive load
        recovery_hours=0.0,       # No recovery needed
        importance_score=0.4,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    request_low_cognitive = ScheduleEventRequest(
        title="Email Processing",
        description="Check and respond to emails",
        duration=0.5,
        preferred_date=datetime.now() + timedelta(days=1)
    )
    
    try:
        async def mock_find_similar_low(user_id, query):
            return MockSimilarResult(low_cognitive_task, 0.8)
        
        task_service.find_similar_task_type = mock_find_similar_low
        
        result_low = await scheduler_service.schedule_with_behavioral_patterns(
            user_id=test_user_id,
            user_energy_pattern=user_energy_pattern,
            request=request_low_cognitive,
            existing_events=[]
        )
        
        # Restore original method
        task_service.find_similar_task_type = original_method
        
        print(f"✅ Low-cognitive task scheduled!")
        print(f"📅 Scheduled: {result_low['event']['scheduled_start'].strftime('%A %H:%M')}")
        print(f"📊 Priority score: {result_low['event']['calculated_priority']:.3f}")
        
        if 'optimal_slot' in result_low:
            slot = result_low['optimal_slot']
            print(f"💡 Reasoning: {slot.get('reasoning', 'N/A')}")
            print(f"🎯 Score: {slot.get('score', 0):.3f}")
        
        print()
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        print()
    
    print("🎉 Enhanced behavioral scheduling tests completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_behavioral_scheduling()) 