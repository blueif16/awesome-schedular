#!/usr/bin/env python3
"""
Demo LangGraph Scheduler - Works with Mock Data
Demonstrates the full chatbot experience without requiring database setup
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain imports
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class SchedulerState(TypedDict):
    """State schema for the scheduling chatbot"""
    messages: Annotated[list, add_messages]


@tool
def schedule_task_demo(
    summary: str,
    description: Optional[str] = None,
    duration: float = 1.0,
    importance_score: float = 0.5,
    user_id: str = "demo_user"
) -> Dict[str, Any]:
    """
    Demo scheduling tool that simulates pattern-based scheduling.
    
    Args:
        summary: Task title/summary (required)
        description: Optional task description  
        duration: Task duration in hours (default: 1.0)
        importance_score: Task importance 0.0-1.0 (default: 0.5)
        user_id: User's unique identifier
    
    Returns:
        Dictionary with scheduling result and event details
    """
    try:
        # Simulate scheduling logic with mock data
        import random
        
        # Mock behavioral patterns
        best_hours = [9, 10, 14, 15, 16]  # Typical focus hours
        suggested_hour = random.choice(best_hours)
        
        # Calculate suggested time (next occurrence of suggested hour)
        now = datetime.now()
        suggested_time = now.replace(hour=suggested_hour, minute=0, second=0, microsecond=0)
        
        # If suggested time is in the past, move to next day
        if suggested_time <= now:
            suggested_time += timedelta(days=1)
        
        end_time = suggested_time + timedelta(hours=duration)
        
        # Mock event ID
        event_id = str(uuid.uuid4())[:8]
        
        # Simulate scheduling method selection
        scheduling_methods = ["pattern_based", "llm_semantic", "direct"]
        method = random.choice(scheduling_methods[:2])  # Prefer pattern_based and LLM
        
        return {
            "success": True,
            "event_id": event_id,
            "message": f"✅ Successfully scheduled '{summary}' for {duration} hour(s)",
            "scheduling_method": method,
            "scheduled_start": suggested_time.strftime("%Y-%m-%d %H:%M"),
            "scheduled_end": end_time.strftime("%Y-%m-%d %H:%M"),
            "reasoning": f"Optimal time based on {method} analysis - high energy period at {suggested_hour}:00",
            "confidence": round(random.uniform(0.75, 0.95), 2)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ Error scheduling '{summary}': {str(e)}"
        }


@tool  
def get_schedule_demo(
    user_id: str = "demo_user",
    days: int = 7
) -> Dict[str, Any]:
    """
    Demo tool to show upcoming schedule.
    
    Args:
        user_id: User's unique identifier  
        days: Number of days to show (default: 7)
        
    Returns:
        Dictionary with schedule information
    """
    try:
        # Mock existing events
        mock_events = [
            {
                "title": "Team standup",
                "start": "09:00",
                "end": "09:30",
                "date": "Today"
            },
            {
                "title": "Code review session", 
                "start": "14:00",
                "end": "15:00",
                "date": "Today"
            },
            {
                "title": "Project planning",
                "start": "10:00", 
                "end": "11:30",
                "date": "Tomorrow"
            }
        ]
        
        schedule_summary = "📅 Your Schedule:\n"
        for event in mock_events:
            schedule_summary += f"• {event['date']}: {event['title']} ({event['start']}-{event['end']})\n"
        
        return {
            "success": True,
            "message": schedule_summary,
            "events_count": len(mock_events),
            "free_slots": ["11:00-12:00 Today", "16:00-17:00 Today", "09:00-10:00 Tomorrow"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ Error retrieving schedule: {str(e)}"
        }


class DemoSchedulingChatbot:
    """Demo LangGraph-based scheduling chatbot with mock data"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the chatbot"""
        
        # Initialize LLM if API key provided, otherwise use mock
        if openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-4",
                api_key=openai_api_key,
                temperature=0.3
            )
            self.use_real_llm = True
        else:
            self.llm = None
            self.use_real_llm = False
            print("🔧 Running in DEMO MODE (no OpenAI API key)")
        
        # Define available tools
        self.tools = [schedule_task_demo, get_schedule_demo]
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Initialize graph builder
        graph_builder = StateGraph(SchedulerState)
        
        # Add chatbot node
        def chatbot_node(state: SchedulerState):
            """Main chatbot logic with tool calling capability"""
            if self.use_real_llm:
                # Use real LLM with tools
                llm_with_tools = self.llm.bind_tools(self.tools)
                return {"messages": [llm_with_tools.invoke(state["messages"])]}
            else:
                # Mock LLM response that demonstrates tool calling
                last_message = state["messages"][-1]["content"].lower()
                
                if "schedule" in last_message and ("task" in last_message or "meeting" in last_message):
                    # Mock tool call for scheduling
                    return {"messages": [{
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "mock_call_1",
                            "name": "schedule_task_demo", 
                            "args": {
                                "summary": "Focus session" if "focus" in last_message else "Meeting",
                                "duration": 2.0 if "2 hour" in last_message else 1.0,
                                "importance_score": 0.8
                            }
                        }]
                    }]}
                elif "show" in last_message and "schedule" in last_message:
                    # Mock tool call for viewing schedule
                    return {"messages": [{
                        "role": "assistant", 
                        "content": "",
                        "tool_calls": [{
                            "id": "mock_call_2",
                            "name": "get_schedule_demo",
                            "args": {"days": 7}
                        }]
                    }]}
                else:
                    # Regular chat response
                    return {"messages": [{
                        "role": "assistant",
                        "content": "👋 Hi! I'm your scheduling assistant. I can help you:\n\n" +
                                 "• Schedule tasks and meetings\n" +
                                 "• Show your current schedule\n" +
                                 "• Find optimal time slots\n\n" +
                                 "Try saying: 'Schedule a 2-hour focus session' or 'Show my schedule'"
                    }]}
        
        graph_builder.add_node("chatbot", chatbot_node)
        
        # Add tool node for executing scheduling tools
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        # Define edges
        graph_builder.add_edge(START, "chatbot")
        
        # Conditional edge: use tools if requested, otherwise end
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {"tools": "tools", END: END}
        )
        
        # Return to chatbot after tool execution
        graph_builder.add_edge("tools", "chatbot")
        
        # Compile the graph
        return graph_builder.compile()
    
    def chat(self, user_input: str, user_id: str = "demo_user") -> str:
        """Process a single chat message"""
        
        # Stream the graph execution
        final_response = None
        for event in self.graph.stream({
            "messages": [{"role": "user", "content": user_input}]
        }):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    latest_message = value["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        final_response = latest_message.content
        
        return final_response or "I apologize, but I couldn't process your request."
    
    def interactive_demo(self):
        """Run an interactive demo session"""
        print("🤖 Scheduling Assistant Demo (LangGraph)")
        print("=" * 50)
        
        if not self.use_real_llm:
            print("🔧 Demo Mode: Using mock responses")
            print("📋 Try these example commands:")
            print("  • 'Hello'")
            print("  • 'Schedule a 2-hour focus session'")
            print("  • 'Show my schedule'")
            print("  • 'Schedule a team meeting'")
        else:
            print("🚀 Live Mode: Using OpenAI GPT-4")
            print("📋 You can ask me to schedule anything!")
            
        print("\nType 'quit', 'exit', or 'q' to end.\n")
        
        while True:
            try:
                user_input = input("💬 You: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Demo ended. Thanks for trying the LangGraph scheduler!")
                    break
                
                print("🤖 Assistant: ", end="")
                response = self.chat(user_input)
                print(response)
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the demo chatbot"""
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("⚠️ No OPENAI_API_KEY found in environment")
        print("🔧 Running in DEMO MODE with mock responses\n")
    
    # Initialize and run demo chatbot
    try:
        chatbot = DemoSchedulingChatbot(openai_api_key)
        chatbot.interactive_demo()
    except Exception as e:
        print(f"❌ Failed to initialize demo chatbot: {e}")


if __name__ == "__main__":
    main() 