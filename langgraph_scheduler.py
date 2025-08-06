"""
LangGraph Scheduler Chatbot - Minimal Implementation
Integrates schedule_with_pattern as a tool for conversational scheduling
"""

import os
import json
from datetime import datetime, timedelta
from typing import Annotated, Dict, Any, Optional, List, Tuple
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain imports
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Local imports
from scheduler_service import SchedulerService
from task_type_service import TaskTypeService
from db_service import DatabaseService


class SchedulerState(TypedDict):
    """State schema for the scheduling chatbot"""
    messages: Annotated[list, add_messages]


@tool
def schedule_task_tool(
    user_id: str,
    summary: str,
    description: Optional[str] = None,
    duration: float = 1.0,
    importance_score: float = 0.5,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timezone: Optional[str] = None
) -> Dict[str, Any]:
    """
    Schedule a task using behavioral patterns and energy optimization.
    
    Args:
        user_id: User's unique identifier
        summary: Task title/summary (required)
        description: Optional task description  
        duration: Task duration in hours (default: 1.0)
        importance_score: Task importance 0.0-1.0 (default: 0.5)
        start_time: Optional specific start time (ISO format)
        end_time: Optional specific end time (ISO format)
        timezone: Optional timezone (IANA format)
    
    Returns:
        Dictionary with scheduling result and event details
    """
    try:
        # Initialize services
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            return {
                "success": False,
                "error": "Supabase credentials not configured",
                "message": "Please configure SUPABASE_URL and SUPABASE_ANON_KEY environment variables"
            }
        
        supabase = create_client(supabase_url, supabase_key)
        task_type_service = TaskTypeService(supabase)
        scheduler_service = SchedulerService(task_type_service)
        
        # Get OpenAI client for LLM fallback
        openai_client = ChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Call the scheduling service (sync wrapper for async method)
        import asyncio
        result = asyncio.run(scheduler_service.schedule_with_pattern(
            user_id=user_id,
            start=start_time,
            end=end_time,
            timeZone=timezone,
            summary=summary,
            description=description,
            duration=duration,
            importance_score=importance_score,
            openai_client=openai_client
        ))
        
        if result.get("success"):
            return {
                "success": True,
                "event_id": result.get("eventId"),
                "message": f"✅ Successfully scheduled '{summary}' for {duration} hour(s)",
                "scheduling_method": "pattern_based"
            }
        else:
            return {
                "success": False,
                "error": "Scheduling failed",
                "message": f"❌ Could not schedule '{summary}'"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ Error scheduling '{summary}': {str(e)}"
        }


class SchedulingChatbot:
    """LangGraph-based scheduling chatbot"""
    
    def __init__(self, openai_api_key: str):
        """Initialize the chatbot with OpenAI integration"""
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=openai_api_key,
            temperature=0.3
        )
        
        # Define available tools
        self.tools = [schedule_task_tool]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Initialize graph builder
        graph_builder = StateGraph(SchedulerState)
        
        # Add chatbot node
        def chatbot_node(state: SchedulerState):
            """Main chatbot logic with tool calling capability"""
            return {"messages": [self.llm_with_tools.invoke(state["messages"])]}
        
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
    
    def chat(self, user_input: str, user_id: str = "test_user") -> str:
        """Process a single chat message"""
        
        # Add user context to the message
        enhanced_input = f"[User ID: {user_id}] {user_input}"
        
        # Stream the graph execution
        final_response = None
        for event in self.graph.stream({
            "messages": [{"role": "user", "content": enhanced_input}]
        }):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    latest_message = value["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        final_response = latest_message.content
        
        return final_response or "I apologize, but I couldn't process your request."
    
    def interactive_chat(self):
        """Run an interactive chat session"""
        print("🤖 Scheduling Assistant (LangGraph)")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        print("Example: 'Schedule a 2-hour focus session for tomorrow morning'\n")
        
        user_id = input("Enter your user ID (or press Enter for 'test_user'): ").strip()
        if not user_id:
            user_id = "test_user"
        
        while True:
            try:
                user_input = input(f"\n[{user_id}] You: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                response = self.chat(user_input, user_id)
                print(f"🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the scheduling chatbot"""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for required API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        return
    
    # Initialize and run chatbot
    try:
        chatbot = SchedulingChatbot(openai_api_key)
        chatbot.interactive_chat()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")


if __name__ == "__main__":
    main() 