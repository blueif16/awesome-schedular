#!/usr/bin/env python3
"""
Test script for simplified API endpoints that return only event IDs
"""

import requests
import json
import time

API_BASE = "http://localhost:5000"

def test_direct_schedule():
    """Test direct scheduling endpoint - should return event ID or None"""
    print("🔍 Testing direct scheduling endpoint...")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/schedule",
            json={"task": "Important team meeting"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Direct schedule result: {result}")
            if result:
                print(f"   Event ID: {result}")
            else:
                print("   Result: None (scheduling failed)")
        else:
            print(f"❌ Direct schedule failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Direct schedule error: {e}")

def test_chat_schedule():
    """Test chat scheduling endpoint - should return event ID or None"""
    print("\n🔍 Testing chat scheduling endpoint...")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"message": "Schedule a 2-hour deep work session"},
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat schedule result: {result}")
            if result:
                print(f"   Event ID: {result}")
            else:
                print("   Result: None (scheduling failed)")
        else:
            print(f"❌ Chat schedule failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Chat schedule error: {e}")

def test_chat_non_schedule():
    """Test chat with non-scheduling message"""
    print("\n🔍 Testing chat with non-scheduling message...")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"message": "Hello, what can you help me with?"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Non-schedule chat result: {result}")
        else:
            print(f"❌ Non-schedule chat failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Non-schedule chat error: {e}")

def main():
    print("🧪 Simplified API Endpoint Tests")
    print("=" * 50)
    print("📝 Both endpoints now return only event ID (string) or None")
    print("   No more success/error object structures\n")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API health check failed")
            return
        print("✅ API is running\n")
    except:
        print("❌ API is not running. Start it with: cd api && python app.py")
        return
    
    # Run tests
    test_direct_schedule()
    time.sleep(2)
    
    test_chat_schedule()
    time.sleep(2)
    
    test_chat_non_schedule()
    
    print("\n🏁 Simplified endpoint tests completed!")
    print("\n📋 Expected responses:")
    print("   • Success: event ID string (e.g., 'abc123-def456-789')")
    print("   • Failure: null")
    print("   • Frontend can now easily check: if (response) { success } else { failed }")

if __name__ == "__main__":
    main() 