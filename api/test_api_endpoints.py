#!/usr/bin/env python3
"""
Test script for the two scheduling endpoints:
1. /api/schedule - Direct pattern-based scheduling
2. /api/chat - LangGraph conversational scheduling
"""

import requests
import json
import time

API_BASE = "http://localhost:5000"

def test_health_check():
    """Test the health endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_direct_scheduling():
    """Test the direct scheduling endpoint"""
    print("\n🔍 Testing direct scheduling endpoint (/api/schedule)...")
    
    test_cases = [
        {
            "task": "Team meeting",
            "duration": 1.0,
            "importance_score": 0.7
        },
        {
            "task": "Deep work session",
            "duration": 2.0,
            "importance_score": 0.9
        },
        {
            "task": "Email cleanup",
            "duration": 0.5,
            "importance_score": 0.3
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['task']}")
        try:
            response = requests.post(
                f"{API_BASE}/api/schedule",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ Scheduled successfully: {result.get('eventId', 'No ID')}")
                    print(f"   Method: {result.get('scheduling_method', 'Unknown')}")
                else:
                    print(f"❌ Scheduling failed: {result.get('message', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request error: {e}")

def test_chat_scheduling():
    """Test the LangGraph chat endpoint"""
    print("\n🔍 Testing LangGraph chat endpoint (/api/chat)...")
    
    chat_messages = [
        "Hello! Can you help me schedule tasks?",
        "I need to schedule a 2-hour focus session for tomorrow",
        "Can you schedule a team standup for 30 minutes?",
        "Thanks for your help!"
    ]
    
    for i, message in enumerate(chat_messages, 1):
        print(f"\n💬 Chat {i}: {message}")
        try:
            response = requests.post(
                f"{API_BASE}/api/chat",
                json={"message": message},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    chat_response = result.get('response', 'No response')
                    print(f"🤖 Response: {chat_response[:150]}...")
                    if len(chat_response) > 150:
                        print("    [truncated]")
                else:
                    print(f"❌ Chat failed: {result.get('message', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request error: {e}")
        
        # Small delay between chat messages
        time.sleep(1)

def test_status_endpoint():
    """Test the status endpoint"""
    print("\n🔍 Testing status endpoint (/api/status)...")
    try:
        response = requests.get(f"{API_BASE}/api/status")
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                patterns = result.get('patterns', [])
                print(f"✅ Status retrieved: {len(patterns)} task types found")
                for pattern in patterns[:3]:  # Show first 3
                    print(f"   • {pattern.get('task_type', 'Unknown')}: {pattern.get('completion_count', 0)} completions")
            else:
                print(f"❌ Status failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Request error: {e}")

def main():
    """Run all tests"""
    print("🧪 API Endpoint Tests")
    print("=" * 50)
    
    # Test if API is running
    if not test_health_check():
        print("\n❌ API is not running. Start it with:")
        print("   cd api && python app.py")
        return
    
    # Test all endpoints
    test_direct_scheduling()
    test_chat_scheduling()
    test_status_endpoint()
    
    print("\n🏁 Tests completed!")
    print("\n💡 Usage examples:")
    print("Direct scheduling:")
    print('   curl -X POST http://localhost:5000/api/schedule -H "Content-Type: application/json" -d \'{"task": "Review code", "duration": 1.5}\'')
    print("\nChat scheduling:")
    print('   curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d \'{"message": "Schedule a meeting for tomorrow"}\'')

if __name__ == "__main__":
    main() 