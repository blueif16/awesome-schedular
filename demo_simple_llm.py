#!/usr/bin/env python3
"""
Simple Demo: LLM Task Analysis
Shows how the system analyzes task characteristics using OpenAI GPT-4
"""

import os
import json
import asyncio
import openai

async def analyze_task_characteristics(openai_client, task_type: str, description: str = None):
    """Use LLM to analyze task characteristics: cognitive load, importance, duration, and recovery time"""
    
    # Prepare task context for analysis
    task_context = f"Task: {task_type}"
    if description:
        task_context += f"\nDescription: {description}"
    
    function_schema = {
        "name": "analyze_task_characteristics",
        "description": "Analyze task characteristics to determine cognitive load, importance, typical duration, and recovery time",
        "parameters": {
            "type": "object",
            "properties": {
                "cognitive_load": {
                    "type": "number",
                    "description": "Mental difficulty/complexity (0.0-1.0). 0.1=routine/automatic, 0.3=simple focus, 0.5=moderate thinking, 0.7=complex analysis, 0.9=deep concentration/creativity",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "importance_score": {
                    "type": "number", 
                    "description": "Task importance/priority (0.0-1.0). 0.1=trivial, 0.3=nice-to-have, 0.5=normal, 0.7=important, 0.9=critical/urgent",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "typical_duration": {
                    "type": "number",
                    "description": "Estimated typical duration in hours (0.25-8.0). Consider task complexity and standard practices",
                    "minimum": 0.25,
                    "maximum": 8.0
                },
                "recovery_hours": {
                    "type": "number",
                    "description": "Buffer/recovery time needed after task in hours (0.0-2.0). Higher for mentally draining tasks",
                    "minimum": 0.0,
                    "maximum": 2.0
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the analysis rationale"
                }
            },
            "required": ["cognitive_load", "importance_score", "typical_duration", "recovery_hours", "reasoning"]
        }
    }
    
    analysis_prompt = f"""
Analyze this task and determine its characteristics:

{task_context}

Consider:
- Cognitive Load: How much mental effort/concentration is required?
- Importance: How critical is this task typically? (not urgent, but inherent importance)
- Typical Duration: How long does this type of task usually take?
- Recovery Time: How much buffer time is needed after this task?

Examples:
- "Email checking": cognitive_load=0.2, importance=0.4, duration=0.5h, recovery=0.1h
- "Strategic planning": cognitive_load=0.8, importance=0.8, duration=2.0h, recovery=0.5h  
- "Client presentation": cognitive_load=0.6, importance=0.7, duration=1.5h, recovery=0.3h
- "Code review": cognitive_load=0.7, importance=0.6, duration=1.0h, recovery=0.2h
"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an expert task analyst. Analyze tasks to determine their cognitive requirements, typical importance, duration, and recovery needs. Be precise and consider real-world work patterns."
            },
            {
                "role": "user", 
                "content": analysis_prompt
            }
        ],
        functions=[function_schema],
        function_call={"name": "analyze_task_characteristics"},
        temperature=0.3
    )
    
    function_call = response.choices[0].message.function_call
    if function_call and function_call.name == "analyze_task_characteristics":
        result = json.loads(function_call.arguments)
        
        # Validate and clamp values
        return {
            "cognitive_load": max(0.0, min(1.0, result.get("cognitive_load", 0.5))),
            "importance_score": max(0.0, min(1.0, result.get("importance_score", 0.5))),
            "typical_duration": max(0.25, min(8.0, result.get("typical_duration", 1.0))),
            "recovery_hours": max(0.0, min(2.0, result.get("recovery_hours", 0.5))),
            "reasoning": result.get("reasoning", "LLM analysis completed")
        }
    else:
        raise ValueError("LLM did not return expected function call")

async def main():
    print("🧠 LLM Task Analysis Demo")
    print("=" * 60)
    
    # Setup OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("💡 Tip: export OPENAI_API_KEY=your_key_here")
        return
    
    openai_client = openai.OpenAI(api_key=openai_api_key)
    
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
            "title": "Email Cleanup",
            "description": "Organize inbox and respond to non-urgent emails"
        }
    ]
    
    print(f"📋 Analyzing {len(test_tasks)} different task types...\n")
    
    # Analyze each task
    for i, task in enumerate(test_tasks, 1):
        print(f"📋 Task {i}: {task['title']}")
        print(f"   📝 Description: {task['description']}")
        
        try:
            # Analyze using LLM
            analysis = await analyze_task_characteristics(
                openai_client, task['title'], task['description']
            )
            
            # Display results
            cognitive_label = 'Low' if analysis['cognitive_load'] < 0.4 else 'Medium' if analysis['cognitive_load'] < 0.7 else 'High'
            importance_label = 'Low' if analysis['importance_score'] < 0.4 else 'Medium' if analysis['importance_score'] < 0.7 else 'High'
            
            print(f"   🧠 LLM Analysis Results:")
            print(f"      🎯 Cognitive Load: {analysis['cognitive_load']:.2f} ({cognitive_label})")
            print(f"      ⭐ Importance: {analysis['importance_score']:.2f} ({importance_label})")
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
    print("\n✅ Task types created with these characteristics will have smart")
    print("   scheduling that considers cognitive load and energy patterns!")

if __name__ == "__main__":
    asyncio.run(main()) 