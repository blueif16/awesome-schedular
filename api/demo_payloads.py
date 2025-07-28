#!/usr/bin/env python3
"""
Demo payloads for the Smart Scheduler API
Shows exact request formats for frontend integration
"""

import json
from datetime import datetime

def print_demo_payloads():
    """Print example API request payloads"""
    
    print("🎯 Smart Scheduler API - Request Examples")
    print("=" * 50)
    print()
    
    # Chat message endpoint
    chat_payload = {
        "content": "Schedule a meeting with the team",
        "timestamp": datetime.now().isoformat(),
        "userId": "any-user-will-be-ignored"
    }
    
    print("📋 POST /api/chat/message")
    print(json.dumps(chat_payload, indent=2))
    print()
    
    # Onboarding endpoint  
    onboarding_payload = {
        "preferences": ["Morning exercise", "Learn Spanish"],
        "timestamp": datetime.now().isoformat(),
        "userId": "this-will-also-be-ignored"
    }
    
    print("👋 POST /api/onboarding")
    print(json.dumps(onboarding_payload, indent=2))
    print()
    
    print("🔑 Key Features:")
    print("✅ Always uses default email: 'x'")
    print("✅ Always uses role: 'student'") 
    print("✅ Ignores any provided userId")
    print("✅ Returns exact BackendEventResponse format")
    print("✅ Returns exact OnboardingResponse format")
    print()
    
    print("🚀 To start the API:")
    print("python app.py")
    print()
    
    print("🧪 To test the API:")
    print("python test_api.py")

if __name__ == "__main__":
    print_demo_payloads() 