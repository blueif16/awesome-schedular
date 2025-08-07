#!/usr/bin/env python3
"""
Demo: LLM Task Analysis
Shows how the system analyzes different task types using LLM
"""

import os
import sys
import asyncio
from supabase import create_client
sys.path.append('..')
from task_type_service import TaskTypeService
from uuid import uuid4

async def main():
    print("🧠 LLM Task Analysis Demo")
    print("=" * 60)
    
    # Setup
    supabase = create_client(
        os.getenv("SUPABASE_URL"), 
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    openai_api_key = os.getenv("OPENAI_API_KEY")
    task_service = TaskTypeService(supabase, openai_api_key)
    
    # Create test user
    test_user_id = str(uuid4())
    
    # Test different task types
    test_tasks = [
        {
            "title": "Weekly Team Standup",
            "description": "Quick sync meeting with development team"
        },
        {
            "title": "Strategic Product Planning",
            "description": "Deep analysis of market opportunities and product roadmap for next quarter"
        },
        {
            "title": "Code Review",
            "description": "Review pull requests and provide technical feedback"
        },
        {
            "title": "Social Media Posting",
            "description": "Quick post updates on company social accounts"
        },
        {
            "title": "Client Presentation Prep",
            "description": "Prepare slides and talking points for important client pitch"
        },
        {
            "title": "Email Cleanup",
            "description": "Organize inbox and respond to non-urgent emails"
        }
    ]
    
    print(f"👤 Testing with user: {test_user_id[:8]}...\n")
    
    # Analyze each task
    for i, task in enumerate(test_tasks, 1):
        print(f"📋 Task {i}: {task['title']}")
        print(f"   📝 Description: {task['description']}")
        
        try:
            # Analyze using LLM
            analysis = await task_service.analyze_task_characteristics(
                task['title'], task['description']
            )
            
            # Display results
            print(f"   🧠 LLM Analysis Results:")
            print(f"      🎯 Cognitive Load: {analysis['cognitive_load']:.2f} ({'Low' if analysis['cognitive_load'] < 0.4 else 'Medium' if analysis['cognitive_load'] < 0.7 else 'High'})")
            print(f"      ⭐ Importance: {analysis['importance_score']:.2f} ({'Low' if analysis['importance_score'] < 0.4 else 'Medium' if analysis['importance_score'] < 0.7 else 'High'})")
            print(f"      ⏱️  Duration: {analysis['typical_duration']:.1f} hours")
            print(f"      🔄 Recovery: {analysis['recovery_hours']:.1f} hours")
            print(f"      💭 Reasoning: {analysis['reasoning']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("🎯 Analysis Summary:")
    print("   • High cognitive tasks (0.7+): Strategic planning, complex analysis")  
    print("   • Medium cognitive tasks (0.4-0.7): Code review, presentations")
    print("   • Low cognitive tasks (0.0-0.4): Email, social media, quick meetings")
    print("   • High importance tasks (0.7+): Client work, strategic planning")
    print("   • Recovery time scales with cognitive load (mental fatigue)")

if __name__ == "__main__":
    asyncio.run(main()) 