from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path

# Ensure project root is on sys.path so absolute imports work when run as script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import get_mem0_service  # type: ignore

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

# Configure logging - improved to respect environment variables
log_dir = os.path.join("core", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "mem0_mcp.log")

# Get log level from environment variable, default to INFO
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Create a logger for this module specifically
logger = logging.getLogger("mem0_mcp")
logger.setLevel(log_level)

# Check if handlers already exist to avoid duplicates
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Console handler: direct to stderr to avoid interfering with MCP stdio protocol
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info(f"Mem0 MCP server initializing with log level: {log_level_name}")

# Initialize FastMCP server
mcp = FastMCP("mem0")
logger.info("Mem0 MCP server initialized")

# Get mem0 service instance
mem0_service = get_mem0_service()
logger.info("Mem0 service instance obtained")


@mcp.tool()
async def extract_and_store_text_insights(
    user_id: str, 
    text_content: str,
    context: str = "conversation"
) -> str:
    """
    LLM-powered extraction and categorization of user content into 4 memory categories.
    Store user preferences, insights, and non-scheduling information using LLM categorization.

    Args:
        user_id (str): The user's unique identifier
        text_content (str): User's statement or preference to store
        context (str): Context of the conversation. Defaults to "conversation".
        
    Returns:
        str: Brief message about storage results indicating which categories were updated
    """
    
    logger.info("MEM0 MCP TOOL INVOKED")
    logger.info(f"MEM0 PARAMS: user_id={user_id}, context='{context}'")
    logger.info(f"MEM0 PARAMS: text_content='{text_content}'")
    
    try:
        # Call the mem0 service class method
        result = await mem0_service.extract_and_store_text_insights(
            user_id=user_id,
            text_content=text_content,
            context=context
        )
        logger.info("Mem0 service completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Mem0 service failed: {e}", exc_info=True)
        return f"Memory storage failed: {str(e)}"


if __name__ == "__main__":
    # 确保环境变量设置
    if not os.getenv("MASTER_ENCRYPTION_KEY"):
        os.environ["MASTER_ENCRYPTION_KEY"] = "solara-default-key-2024"
    
    # Run MCP server
    logger.info("Starting Mem0 MCP server with stdio transport")
    try:
        mcp.run(transport="stdio")
        logger.info("Mem0 MCP server running")
    except Exception as e:
        logger.error(f"Failed to start Mem0 MCP server: {e}")
        import traceback
        logger.error(traceback.format_exc()) 