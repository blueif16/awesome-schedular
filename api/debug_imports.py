#!/usr/bin/env python3
"""
Debug script to test imports and initialization
"""

import os
import sys
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 Testing imports...")

try:
    from dotenv import load_dotenv
    print("✅ dotenv imported")
except Exception as e:
    print(f"❌ dotenv import failed: {e}")

try:
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
    print("✅ .env loaded")
except Exception as e:
    print(f"❌ .env load failed: {e}")

try:
    from supabase import create_client, Client
    print("✅ supabase imported")
except Exception as e:
    print(f"❌ supabase import failed: {e}")

try:
    from task_type_service import TaskTypeService
    print("✅ task_type_service imported")
except Exception as e:
    print(f"❌ task_type_service import failed: {e}")
    traceback.print_exc()

try:
    from scheduler_service import SchedulerService
    print("✅ scheduler_service imported")
except Exception as e:
    print(f"❌ scheduler_service import failed: {e}")
    traceback.print_exc()

try:
    from langchain_openai import ChatOpenAI
    print("✅ langchain_openai imported")
except Exception as e:
    print(f"❌ langchain_openai import failed: {e}")
    traceback.print_exc()

try:
    from langgraph.prebuilt import create_react_agent
    print("✅ langgraph imported")
except Exception as e:
    print(f"❌ langgraph import failed: {e}")
    traceback.print_exc()

print("\n🔍 Testing environment variables...")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"SUPABASE_URL: {'✅' if supabase_url else '❌'}")
print(f"SUPABASE_SERVICE_ROLE_KEY: {'✅' if supabase_key else '❌'}")
print(f"OPENAI_API_KEY: {'✅' if openai_api_key else '❌'}")

if not all([supabase_url, supabase_key, openai_api_key]):
    print("❌ Missing required environment variables")
    exit(1)

print("\n🔍 Testing service initialization...")

try:
    supabase = create_client(supabase_url, supabase_key)
    print("✅ Supabase client created")
except Exception as e:
    print(f"❌ Supabase client creation failed: {e}")
    traceback.print_exc()
    exit(1)

try:
    task_type_service = TaskTypeService(supabase, openai_api_key)
    print("✅ TaskTypeService created")
except Exception as e:
    print(f"❌ TaskTypeService creation failed: {e}")
    traceback.print_exc()
    exit(1)

try:
    scheduler_service = SchedulerService(task_type_service)
    print("✅ SchedulerService created")
except Exception as e:
    print(f"❌ SchedulerService creation failed: {e}")
    traceback.print_exc()
    exit(1)

try:
    llm = ChatOpenAI(
        model="gpt-4",
        api_key=openai_api_key,
        temperature=0.3
    )
    print("✅ ChatOpenAI created")
except Exception as e:
    print(f"❌ ChatOpenAI creation failed: {e}")
    traceback.print_exc()

print("\n🔍 Testing async function...")

import asyncio

async def test_schedule():
    try:
        # Use the existing demo user ID that's in the database
        demo_user_id = "33a07e45-c5a8-4b95-9e39-c12752012e36"
        print(f"Using existing demo user_id: {demo_user_id}")
        
        result = await scheduler_service.schedule_with_pattern(
            user_id=demo_user_id,
            summary="Test task",
            openai_client=task_type_service.openai_client
        )
        print(f"✅ schedule_with_pattern result: {result}")
        return result
    except Exception as e:
        print(f"❌ schedule_with_pattern failed: {e}")
        traceback.print_exc()
        return None

# Test async function
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(test_schedule())
loop.close()

print(f"\n🎯 Final result: {result}")
print("✅ All tests completed!") 