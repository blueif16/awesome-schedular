"""
LangGraph Scheduler Chatbot - Enhanced React Agent Implementation
Uses MCP client to connect to scheduler and mem0 servers
"""

import os
import logging
from datetime import datetime
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LangGraph and MCP imports
from langgraph.prebuilt import create_react_agent
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient

logger.info("Setting up MCP integration...")
logger.info("MCP imports loaded successfully")


def get_mcp_config() -> dict:
    """Build MCP server configuration for stdio transport.
    Uses PYTHONPATH if provided to resolve absolute paths, otherwise resolves relative to this file.
    """
    current_dir = Path(__file__).parent.absolute()
    # Prefer absolute paths near this file
    scheduler_path = str((current_dir / "scheduler_mcp.py").resolve())
    mem0_path = str((current_dir / "mem0_mcp.py").resolve())

    return {
        "scheduler": {
            "command": sys.executable,
            "args": [scheduler_path],
            "transport": "stdio",
            "env": {},
        },
        "mem0": {
            "command": sys.executable,
            "args": [mem0_path],
            "transport": "stdio",
            "env": {},
        },
    }


class SchedulingChatbot:
    """Enhanced LangGraph-based scheduling chatbot with MCP integration"""
    
    def __init__(self):
        """Initialize the chatbot with MCP client"""
        logger.info("Initializing SchedulingChatbot with MCP...")
        
        # Setup MCP client with our servers
        self.client = MultiServerMCPClient(get_mcp_config())
        
        logger.info("MCP client initialized with scheduler and mem0 servers")
        
        # Initialize OpenAI model name
        self.model_name = "gpt-4o"
        logger.info(f"Using model: {self.model_name}")
        
        # Will be set in initialize()
        self.tools = None
        self.agent = None
    
    async def initialize(self):
        """Initialize MCP tools and create agent"""
        logger.info("Getting tools from MCP servers...")
        
        try:
            # Get tools from all connected MCP servers
            self.tools = await self.client.get_tools()
            logger.error(f"Retrieved {len(self.tools)} tools from MCP servers")
            
            # Log available tools
            for tool in self.tools:
                logger.info(f"Available tool: {tool.name}")
            
            # Create React agent with MCP tools
            self.agent = create_react_agent(self.model_name, self.tools, version="v1")
            logger.info("React agent created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}", exc_info=True)
            raise
    
    async def chat(self, user_input: str, user_id: str = "test_user") -> str:
        """Process a single chat message using MCP-powered React agent"""
        
        logger.info(f"Chat request from user {user_id}: '{user_input}'")
        
        # Ensure agent is initialized
        if not self.agent:
            await self.initialize()
        
        # Get current date/time for context
        current_time = datetime.now()
        current_date_str = current_time.strftime("%A, %B %d, %Y")
        current_time_str = current_time.strftime("%I:%M %p")
        
        # Enhanced input with context
        enhanced_input = f"""Current context:
Today is: {current_date_str}
Current time: {current_time_str}
User ID: {user_id}

User request: {user_input}

Instructions: 
- For scheduling requests, use schedule_with_pattern tool
- For storing preferences/insights, use extract_and_store_text_insights tool
- Consider current date/time when interpreting relative references like "tomorrow", "next week"
"""
        
        try:
            logger.info("Invoking MCP-powered React agent...")
            
            # Invoke the react agent with MCP tools
            response = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": enhanced_input}]
            })
            
            logger.debug(f"Raw agent response: {response}")
            
            # Extract the final message content
            if response and "messages" in response:
                final_message = response["messages"][-1]
                logger.info(f"Agent returned {len(response['messages'])} messages")
                
                if hasattr(final_message, 'content'):
                    content = final_message.content
                    logger.info(f"Extracted content: {content[:100]}...")
                    return content
                elif isinstance(final_message, dict) and "content" in final_message:
                    content = final_message["content"]
                    logger.info(f"Extracted content from dict: {content[:100]}...")
                    return content
                else:
                    logger.warning(f"Unexpected message format: {type(final_message)}")
            else:
                logger.error("No messages in agent response")
            
            return "I apologize, but I couldn't process your request."
            
        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
    
    async def interactive_chat(self):
        """Run an interactive chat session with MCP-powered capabilities"""
        print("MCP-Powered Scheduling Assistant")
        print("I can help you schedule tasks AND remember your preferences using MCP servers!")
        print("Examples:")
        print("  - 'Schedule a 2-hour focus session tomorrow morning'")
        print("  - 'I'm most productive in the morning'")
        print("  - 'I hate working late at night'")
        print("Type 'quit', 'exit', or 'q' to end the conversation.\n")
        
        user_id = input("Enter your user ID (or press Enter for default): ").strip()
        if not user_id:
            user_id = "33a07e45-c5a8-4b95-9e39-c12752012e36"
        
        # Initialize MCP tools
        await self.initialize()
        
        while True:
            try:
                user_input = input(f"\n[{user_id}] You: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print(" Goodbye!")
                    break
                
                response = await self.chat(user_input, user_id)
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")


async def main():
    """Main function to run the MCP-powered scheduling chatbot"""
    
    try:
        chatbot = SchedulingChatbot()
        await chatbot.interactive_chat()
    except Exception as e:
        logger.error(f"Failed to run chatbot: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 