#!/usr/bin/env python3
"""
Quick test to verify the updated API with direct schedule_with_pattern tool integration
"""

import requests
import json

def test_health():
    """Quick health check"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not reachable: {e}")
        return False

def test_direct_schedule():
    """Test direct scheduling endpoint"""
    print("\n🔍 Testing direct scheduling...")
    try:
        response = requests.post(
            "http://localhost:5000/api/schedule",
            json={
                "task": "Test meeting",
                "duration": 1.0,
                "importance_score": 0.7
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Direct schedule success: {result.get('success', False)}")
            if result.get('eventId'):
                print(f"   Event ID: {result['eventId']}")
        else:
            print(f"❌ Direct schedule failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"❌ Direct schedule error: {e}")

def test_chat():
    """Test chat endpoint"""
    print("\n🔍 Testing chat endpoint...")
    try:
        response = requests.post(
            "http://localhost:5000/api/chat",
            json={"message": "Hello, can you schedule a 1-hour meeting?"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat success: {result.get('success', False)}")
            if result.get('response'):
                print(f"   Response: {result['response'][:100]}...")
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"❌ Chat error: {e}")

def main():
    print("🧪 Quick API Test")
    print("=" * 30)
    
    if not test_health():
        print("\n💡 Start the API with: cd api && python app.py")
        return
    
    test_direct_schedule()
    test_chat()
    
    print("\n🏁 Quick test completed!")

if __name__ == "__main__":
    main() 