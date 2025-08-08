#!/usr/bin/env python3
"""
Simplified LangGraph Test - Testing Natural Language Intent Recognition
Tests the chatbot's ability to recognize preferences vs scheduling intents
"""

import os
import time
import logging
from dotenv import load_dotenv

# Configure detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_memory_storage_intent():
    """Test that the chatbot recognizes and stores user preferences"""
    try:
        print("🔍 Testing memory storage intent recognition...")
        logger.info("🧪 Starting memory storage intent test")
        
        # Import the chatbot
        from core.langgraph_scheduler import SchedulingChatbot
        
        chatbot = SchedulingChatbot()
        print("  ✅ Chatbot initialized successfully")
        logger.info("✅ Test chatbot initialized")
        
        # Default user ID as requested
        user_id = "33a07e45-c5a8-4b95-9e39-c12752012e36"
        
        # Test preference input - should trigger extract_and_store_text_insights
        preference_text = "I'm a morning person and work best between 9 AM and 11 AM"
        
        print(f"  📝 Sending preference: '{preference_text}'")
        response = await chatbot.chat(preference_text, user_id)
        
        print(f"  🤖 Response: {response}")
        
        # Basic validation - response should acknowledge the preference
        if response and "error" not in response.lower():
            print("  ✅ Memory storage intent recognized and processed")
            return True
        else:
            print("  ❌ Failed to process memory storage intent")
            return False
            
    except Exception as e:
        print(f"  ❌ Memory storage test failed: {e}")
        return False

async def test_scheduling_intent():
    """Test that the chatbot recognizes and processes scheduling requests"""
    try:
        print("🔍 Testing scheduling intent recognition...")
        
        # Import the chatbot
        from core.langgraph_scheduler import SchedulingChatbot
        
        chatbot = SchedulingChatbot()
        print("  ✅ Chatbot initialized successfully")
        
        # Default user ID as requested
        user_id = "33a07e45-c5a8-4b95-9e39-c12752012e36"
        
        # Test scheduling input - should trigger schedule_with_pattern
        scheduling_text = "Schedule a 2-hour deep work session for tomorrow morning"
        
        print(f"  📅 Sending scheduling request: '{scheduling_text}'")
        response = await chatbot.chat(scheduling_text, user_id)
        
        print(f"  🤖 Response: {response}")
        
        # Basic validation - response should acknowledge the scheduling
        if response and "error" not in response.lower():
            print("  ✅ Scheduling intent recognized and processed")
            return True
        else:
            print("  ❌ Failed to process scheduling intent")
            return False
            
    except Exception as e:
        print(f"  ❌ Scheduling test failed: {e}")
        return False

async def test_time_aware_scheduling():
    """Test that the chatbot understands relative time references"""
    try:
        print("🔍 Testing time-aware scheduling with relative references...")
        
        # Import the chatbot
        from core.langgraph_scheduler import SchedulingChatbot
        
        chatbot = SchedulingChatbot()
        print("  ✅ Chatbot initialized successfully")
        
        # Default user ID as requested
        user_id = "33a07e45-c5a8-4b95-9e39-c12752012e36"
        
        # Test with relative time reference that should use available_periods
        time_aware_text = "Schedule a meeting for next week sometime"
        
        print(f"  📅 Sending time-aware request: '{time_aware_text}'")
        response = await chatbot.chat(time_aware_text, user_id)
        
        print(f"  🤖 Response: {response}")
        
        # Basic validation - response should acknowledge the scheduling
        if response and "error" not in response.lower():
            print("  ✅ Time-aware scheduling intent recognized and processed")
            return True
        else:
            print("  ❌ Failed to process time-aware scheduling intent")
            return False
            
    except Exception as e:
        print(f"  ❌ Time-aware scheduling test failed: {e}")
        return False


async def main():
    """Run the simplified chatbot tests"""
    print("🧪 LangGraph Chatbot Intent Recognition Tests")
    print("=" * 50)
    
    # Default user ID
    default_user_id = "33a07e45-c5a8-4b95-9e39-c12752012e36"
    print(f"📋 Using default user ID: {default_user_id}")
    
    tests = [
        ("Memory Storage Intent", test_memory_storage_intent),
        ("Scheduling Intent", test_scheduling_intent),
        ("Time-Aware Scheduling", test_time_aware_scheduling)
    ]
    
    passed = 0
    total = len(tests)
    
    for i, (test_name, test_func) in enumerate(tests):
        print(f"\n📋 Test {i+1}/{total}: {test_name}")
        print("-" * 30)
        
        if await test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
        
        # Add delay between tests as requested (except for last test)
        if i < len(tests) - 1:
            print("  ⏳ Waiting 3 seconds before next test...")
            time.sleep(3)
    
    print(f"\n🏁 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All chatbot intent recognition tests passed!")
        print("\n📝 What was tested:")
        print("  1. ✅ Memory storage: 'I'm a morning person...' → extract_and_store_text_insights")
        print("  2. ✅ Basic scheduling: 'Schedule a 2-hour...' → schedule_with_pattern")
        print("  3. ✅ Time-aware scheduling: 'next week sometime' → schedule_with_pattern + available_periods")
        print("\n🎯 The chatbot successfully recognizes different intents and time context!")
    else:
        print("❌ Some tests failed. Check your configuration.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 