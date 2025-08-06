#!/usr/bin/env python3
"""
Test script to verify comprehensive logging in both LangGraph and scheduler service
"""

import requests
import json
import time

API_BASE = "http://localhost:5000"

def test_direct_schedule_with_logging():
    """Test direct scheduling to see scheduler service logs"""
    print("🔍 Testing direct scheduling endpoint with logging...")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/schedule",
            json={
                "task": "Important team meeting",
                "duration": 1.5,
                "importance_score": 0.8
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Direct schedule result: {result}")
        else:
            print(f"❌ Direct schedule failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Direct schedule error: {e}")

def test_chat_with_logging():
    """Test chat endpoint to see LangGraph logs"""
    print("\n🔍 Testing chat endpoint with logging...")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"message": "Schedule a 2-hour deep work session for tomorrow"},
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat result: {result.get('success', False)}")
            if result.get('response'):
                print(f"   Response: {result['response'][:150]}...")
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Chat error: {e}")

def test_simple_chat():
    """Test simple chat without tool calls"""
    print("\n🔍 Testing simple chat (no tool calls)...")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"message": "Hello, what can you help me with?"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Simple chat result: {result.get('success', False)}")
        else:
            print(f"❌ Simple chat failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Simple chat error: {e}")

def main():
    print("🧪 Logging Test Suite")
    print("=" * 40)
    print("📝 This will generate comprehensive logs in both console and scheduler_api.log")
    print("   Check the running API terminal for detailed logs\n")
    
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
    
    # Run tests with pauses to see logs clearly
    test_direct_schedule_with_logging()
    time.sleep(2)
    
    test_simple_chat()
    time.sleep(2)
    
    test_chat_with_logging()
    
    print("\n🏁 Logging tests completed!")
    print("📋 Check the following for logs:")
    print("   • Console output from the running API server")
    print("   • api/scheduler_api.log file")
    print("   • Look for emojis: 📅 🤖 🔧 🔀 🔗 💬 🌐")

if __name__ == "__main__":
    main() 