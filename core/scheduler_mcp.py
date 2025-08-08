from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path

# Ensure project root is on sys.path so absolute imports work when run as script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import get_scheduler_service  # type: ignore

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

# Configure logging - improved to respect environment variables
log_dir = os.path.join("core", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "scheduler_mcp.log")

# Get log level from environment variable, default to INFO
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Create a logger for this module specifically
logger = logging.getLogger("scheduler_mcp")
logger.setLevel(log_level)

# Check if handlers already exist to avoid duplicates
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Console handler for immediate feedback
    # Send console logs to stderr to avoid interfering with MCP stdio protocol
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info(f"Scheduler MCP server initializing with log level: {log_level_name}")

# Initialize FastMCP server
mcp = FastMCP("scheduler")
logger.info("Scheduler MCP server initialized")

# Get scheduler service instance
scheduler_service = get_scheduler_service()
logger.info("Scheduler service instance obtained")


@mcp.tool()
async def schedule_with_pattern(
    user_id: str,
    start: str | None = None,
    end: str | None = None,
    timeZone: str | None = None,
    summary: str | None = None,
    description: str | None = None,
    location: str | None = None,
    category: str | None = None,
    duration: float = 1.0,
    importance_score: float = 0.5,
    deadline: str | None = None,
    available_periods: str | None = None
) -> str:
    """
    Creates a NEW calendar event with the provided details using pattern-based scheduling.
    Routes to LLM if similarity < 0.4 threshold.

    Args:
        user_id (str): The user's ID to use their specific credentials
        start (str, optional): Event start time in ISO 8601 format. Defaults to None.
        end (str, optional): Event end time in ISO 8601 format. Defaults to None.
        timeZone (str, optional): User timezone as IANA Time Zone name. Defaults to None.
        summary (str, optional): Short title/description of the event. Defaults to None.
        description (str, optional): Detailed description of the event. Defaults to None.
        location (str, optional): Location of the event. Defaults to None.
        category (str, optional): If user provide a start time or a fixed time, the category will be "Event", 
                      else if no time or only a deadline, the category will be "Task". Defaults to None.
        duration (float): Duration in hours. Defaults to 1.0.
        importance_score (float): Task importance 0.0-1.0 (0.0=low priority, 1.0=critical). Defaults to 0.5.
        deadline (str, optional): Optional deadline in ISO 8601 format. Defaults to None.
        available_periods (str, optional): Time periods to search within. Format: "start1,end1;start2,end2" 
                      where each date is in ISO 8601 format. Example: "2024-01-15T09:00:00,2024-01-15T17:00:00;2024-01-16T09:00:00,2024-01-16T17:00:00". Defaults to None.
        
    Returns:
        str: Event ID if successful, or error message if failed
    """
    
    logger.info("SCHEDULER MCP TOOL INVOKED")
    logger.info(f"SCHEDULE START: User {user_id}, task '{summary}', duration {duration}h")
    logger.info(f"SCHEDULE PARAMS: start={start}, end={end}, timezone={timeZone}")
    logger.info(f"SCHEDULE PARAMS: importance={importance_score}, deadline={deadline}")
    logger.info(f"SCHEDULE PARAMS: location='{location}', category='{category}'")
    logger.info(f"SCHEDULE PARAMS: available_periods='{available_periods}'")
    logger.info(f"SCHEDULE PARAMS: description='{description or 'None'}'")
    
    try:
        # Call the scheduler service class method
        result = await scheduler_service.schedule_with_pattern(
            user_id=user_id,
            start=start,
            end=end,
            timeZone=timeZone,
            summary=summary,
            description=description,
            location=location,
            category=category,
            duration=duration,
            importance_score=importance_score,
            deadline=deadline,
            available_periods=available_periods
        )
        logger.info("Scheduler service completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Scheduler service failed: {e}", exc_info=True)
        return f"Scheduling failed: {str(e)}"


if __name__ == "__main__":
    # 确保环境变量设置
    if not os.getenv("MASTER_ENCRYPTION_KEY"):
        os.environ["MASTER_ENCRYPTION_KEY"] = "solara-default-key-2024"
    
    # Run MCP server
    logger.info("Starting Scheduler MCP server with stdio transport")
    try:
        mcp.run(transport="stdio")
        logger.info("Scheduler MCP server running")
    except Exception as e:
        logger.error(f"Failed to start Scheduler MCP server: {e}")
        import traceback
        logger.error(traceback.format_exc()) 