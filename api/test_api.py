#!/usr/bin/env python3
"""
Test the Smart Scheduler API endpoints
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("🩺 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_chat_message():
    """Test chat message endpoint for scheduling"""
    print("💬 Testing chat message endpoint...")
    
    payload = {
        "content": "Schedule a meeting with the team",
        "timestamp": datetime.now().isoformat(),
        "userId": "test-user-123"  # Will be ignored, uses default
    }
    
    response = requests.post(f"{BASE_URL}/api/chat/message", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_onboarding():
    """Test onboarding endpoint"""
    print("👋 Testing onboarding endpoint...")
    
    payload = {
        "preferences": ["Morning exercise", "Learn Spanish"],
        "timestamp": datetime.now().isoformat(),
        "userId": "test-user-123"  # Will be ignored, uses default
    }
    
    response = requests.post(f"{BASE_URL}/api/onboarding", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_demo_status():
    """Test demo status endpoint"""
    print("📊 Testing demo status...")
    response = requests.get(f"{BASE_URL}/demo-status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_multiple_chat_messages():
    """Test multiple different chat messages"""
    print("🔄 Testing multiple chat messages...")
    
    test_messages = [
        "Schedule a workout session",
        "Book a doctor appointment",
        "Plan a team sync meeting",
        "Set up a personal call with mom"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"Test {i}: '{message}'")
        payload = {
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(f"{BASE_URL}/api/chat/message", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('event'):
                event = result['event']
                print(f"✅ Created: {event['title']} - {event['date']} {event['time']}")
                print(f"   Priority: {event['priority']}, Category: {event['category']}")
            else:
                print(f"❌ Failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
        print()

if __name__ == "__main__":
    print("🧪 Smart Scheduler API Test Suite")
    print("=" * 50)
    
    try:
        test_health_check()
        test_onboarding()
        test_chat_message()
        test_multiple_chat_messages()
        test_demo_status()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the API server is running on localhost:5000")
        print("   Run: python api/app.py")
    except Exception as e:
        print(f"❌ Test error: {e}") 