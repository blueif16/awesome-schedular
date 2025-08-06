#!/usr/bin/env python3
"""
Smart Scheduler API Demo
Flask API that reuses existing scheduling services
"""

import os
import sys
import uuid
import json
import asyncio
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client



# Configure logging without emoji for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler_api.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import our services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_type_service import TaskTypeService
from learning_service import LearningService
from scheduler_service import SchedulerService
from hybrid_learning_service import HybridLearningService
from models import UserRole, ScheduleEventRequest, CompleteEventRequest

# LangGraph imports
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Default user credentials - always use these
DEFAULT_EMAIL = "x"
DEFAULT_ROLE = UserRole.STUDENT
DEMO_USER_ID = "33a07e45-c5a8-4b95-9e39-c12752012e36"  # Fixed demo user ID

class ReactSchedulingAgent:
    """React agent for scheduling with simplified output"""
    
    def __init__(self, scheduler_service: SchedulerService, openai_api_key: str, demo_user_id: str):
        """Initialize the React agent with existing services"""
        self.scheduler_service = scheduler_service
        self.demo_user_id = DEMO_USER_ID  # Use fixed demo user ID
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=openai_api_key,
            temperature=0.3
        )
        
        # Create schedule tool
        self.schedule_tool = self._create_schedule_tool()
        
        # Create checkpointer for conversation memory
        self.checkpointer = InMemorySaver()
        
        # Build the React agent
        self.agent = self._build_agent()
    
    def _create_schedule_tool(self):
        """Create schedule tool that returns only event ID"""
        
        @tool
        async def schedule_task(summary: str) -> Optional[str]:
            """
            Schedule a task and return only the event ID.
            
            Args:
                summary: Task title/description (required)
            
            Returns:
                Event ID if successful, None if failed
            """
            try:
                logger.info(f"TOOL_STEP_1: Starting task scheduling process")
                logger.info(f"TOOL_STEP_1: Task summary: '{summary}'")
                logger.info(f"TOOL_STEP_1: Summary length: {len(summary)} characters")
                logger.info(f"TOOL_STEP_1: Target user ID: {self.demo_user_id}")
                
                # Validate prerequisites
                logger.info(f"TOOL_STEP_2: Validating service availability")
                logger.info(f"TOOL_STEP_2: Scheduler service available: {self.scheduler_service is not None}")
                logger.info(f"TOOL_STEP_2: Task type service available: {self.scheduler_service.task_type_service is not None}")
                logger.info(f"TOOL_STEP_2: OpenAI client available: {self.scheduler_service.task_type_service.openai_client is not None}")
                
                # Call the scheduler service
                logger.info("TOOL_STEP_3: Invoking schedule_with_pattern method")
                result = await self.scheduler_service.schedule_with_pattern(
                    user_id=self.demo_user_id,
                    summary=summary,
                    openai_client=self.scheduler_service.task_type_service.openai_client
                )
                
                logger.info(f"TOOL_STEP_3: Schedule_with_pattern completed")
                logger.info(f"TOOL_STEP_3: Raw result type: {type(result)}")
                logger.info(f"TOOL_STEP_3: Raw result keys: {list(result.keys()) if result and isinstance(result, dict) else 'Not a dict'}")
                
                # Process the result
                logger.info(f"TOOL_STEP_4: Processing scheduling result")
                if result:
                    # Result is now just the event ID string
                    event_id = result if isinstance(result, str) else None
                    success = bool(event_id)
                    scheduling_method = 'pattern_based'  # Default method
                    
                    logger.info(f"TOOL_STEP_4: Extracted event ID: {event_id}")
                    logger.info(f"TOOL_STEP_4: Success flag: {success}")
                    logger.info(f"TOOL_STEP_4: Scheduling method: {scheduling_method}")
                    
                    if event_id and success:
                        logger.info(f"TOOL_STEP_4: Scheduling successful - returning event ID: {event_id}")
                        return event_id
                    else:
                        logger.warning(f"TOOL_STEP_4: Scheduling failed - missing event ID or success flag false")
                        return None
                else:
                    logger.warning(f"TOOL_STEP_4: No result returned from scheduler service")
                    return None
                
            except Exception as e:
                logger.error(f"TOOL_ERROR: Scheduling process failed")
                logger.error(f"TOOL_ERROR: Exception type: {type(e).__name__}")
                logger.error(f"TOOL_ERROR: Exception message: {str(e)}")
                import traceback
                logger.error(f"TOOL_ERROR: Full stacktrace: {traceback.format_exc()}")
                return None
        
        return schedule_task
    
    def _build_agent(self):
        """Build React agent with custom prompt for event ID output"""
        
        prompt = """You are a helpful scheduling assistant. When users ask you to schedule something, use the schedule_task tool.

IMPORTANT: After scheduling a task, respond with ONLY the event ID that the tool returns, nothing else. 
If scheduling fails (tool returns None), respond with "None".

Examples:
- User: "Schedule a meeting for tomorrow"
- Tool returns: "abc123"  
- Your response: "abc123"

- User: "Schedule a workout session"
- Tool returns: None
- Your response: "None"

For non-scheduling conversations, respond normally and helpfully."""

        logger.info("AGENT: Creating React agent with schedule tool")
        
        agent = create_react_agent(
            model=self.llm,
            tools=[self.schedule_tool],
            prompt=prompt,
            checkpointer=self.checkpointer
        )
        
        logger.info("AGENT: React agent created successfully")
        return agent
    
    def chat(self, user_input: str, user_id: str = None) -> Optional[str]:
        """Process a single chat message and return event ID or None"""
        
        # Step 1: Initialize user context
        actual_user_id = user_id or self.demo_user_id
        logger.info(f"CHAT_STEP_1: Starting chat session")
        logger.info(f"CHAT_STEP_1: User ID: {actual_user_id}")
        logger.info(f"CHAT_STEP_1: Input message: '{user_input}'")
        logger.info(f"CHAT_STEP_1: Message length: {len(user_input)} characters")
        
        try:
            # Step 2: Configure conversation thread
            config = {"configurable": {"thread_id": actual_user_id}}
            logger.info(f"CHAT_STEP_2: Thread configuration created")
            logger.info(f"CHAT_STEP_2: Thread ID: {actual_user_id}")
            logger.info(f"CHAT_STEP_2: Config structure: {config}")
            
            # Step 3: Prepare agent invocation
            logger.info(f"CHAT_STEP_3: Preparing React agent invocation")
            logger.info(f"CHAT_STEP_3: Agent available: {self.agent is not None}")
            logger.info(f"CHAT_STEP_3: Scheduler service available: {self.scheduler_service is not None}")
            logger.info(f"CHAT_STEP_3: Demo user ID: {self.demo_user_id}")
            
            message_payload = {"messages": [{"role": "user", "content": user_input}]}
            logger.info(f"CHAT_STEP_3: Message payload structure: {message_payload}")
            
            # Step 4: Invoke React agent
            logger.info(f"CHAT_STEP_4: Invoking React agent")
            result = self.agent.invoke(message_payload, config)
            
            logger.info(f"CHAT_STEP_4: Agent invocation completed successfully")
            logger.info(f"CHAT_STEP_4: Result type: {type(result)}")
            logger.info(f"CHAT_STEP_4: Result keys: {list(result.keys()) if result else 'No result'}")
            
            # Step 5: Process agent response
            if result and "messages" in result:
                message_count = len(result['messages'])
                logger.info(f"CHAT_STEP_5: Processing {message_count} messages from agent")
                
                # Step 5a: Log all messages for detailed debugging
                for i, msg in enumerate(result["messages"]):
                    logger.info(f"CHAT_STEP_5a: Message {i+1}/{message_count}")
                    logger.info(f"CHAT_STEP_5a: Message type: {type(msg)}")
                    logger.info(f"CHAT_STEP_5a: Has content attribute: {hasattr(msg, 'content')}")
                    logger.info(f"CHAT_STEP_5a: Has role attribute: {hasattr(msg, 'role')}")
                    logger.info(f"CHAT_STEP_5a: Has tool_calls attribute: {hasattr(msg, 'tool_calls')}")
                    
                    if hasattr(msg, 'content'):
                        content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                        logger.info(f"CHAT_STEP_5a: Message {i+1} content: '{content_preview}'")
                    if hasattr(msg, 'role'):
                        logger.info(f"CHAT_STEP_5a: Message {i+1} role: '{msg.role}'")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        logger.info(f"CHAT_STEP_5a: Message {i+1} tool_calls count: {len(msg.tool_calls) if msg.tool_calls else 0}")
                
                # Step 5b: Extract final response
                final_message = result["messages"][-1]
                logger.info(f"CHAT_STEP_5b: Processing final message")
                logger.info(f"CHAT_STEP_5b: Final message type: {type(final_message)}")
                
                if hasattr(final_message, 'content'):
                    response = final_message.content.strip()
                    logger.info(f"CHAT_STEP_5b: Raw response content: '{response}'")
                    logger.info(f"CHAT_STEP_5b: Response length: {len(response)} characters")
                    
                    # Step 6: Validate and return response
                    logger.info(f"CHAT_STEP_6: Validating response")
                    if response == "None":
                        logger.info("CHAT_STEP_6: Response indicates scheduling failure - returning None")
                        return None
                    elif response and response != "None":
                        logger.info(f"CHAT_STEP_6: Valid event ID received: '{response}'")
                        logger.info(f"CHAT_STEP_6: Event ID format check - length: {len(response)}, alphanumeric: {response.replace('-', '').isalnum()}")
                        return response
                    else:
                        logger.warning("CHAT_STEP_6: Empty or invalid response - returning None")
                        return None
                else:
                    logger.warning("CHAT_STEP_5b: Final message has no content attribute")
                    if hasattr(final_message, '__dict__'):
                        logger.warning(f"CHAT_STEP_5b: Final message attributes: {list(final_message.__dict__.keys())}")
                    return None
            else:
                logger.warning("CHAT_STEP_5: No messages found in agent result")
                if result:
                    logger.warning(f"CHAT_STEP_5: Agent result content: {str(result)[:200]}...")
                return None
        
        except Exception as e:
            logger.error(f"CHAT_ERROR: Agent execution failed")
            logger.error(f"CHAT_ERROR: Error type: {type(e).__name__}")
            logger.error(f"CHAT_ERROR: Error message: {str(e)}")
            import traceback
            logger.error(f"CHAT_ERROR: Full traceback: {traceback.format_exc()}")
            return None

class SmartSchedulerAPI:
    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([supabase_url, supabase_key, openai_api_key]):
            raise ValueError("Missing required environment variables")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_api_key = openai_api_key
        
        # Initialize services
        self.task_type_service = TaskTypeService(self.supabase, openai_api_key)
        self.learning_service = LearningService(self.task_type_service)
        self.scheduler_service = SchedulerService(self.task_type_service)
        
        # Initialize mem0 service if available
        try:
            from mem0_service import get_mem0_service
            self.mem0_service = get_mem0_service()
            if self.mem0_service and not self.mem0_service.is_available:
                asyncio.create_task(self.mem0_service.initialize_memory_service())
            logger.info("API_INIT: Mem0 service initialized successfully")
        except Exception as e:
            logger.warning(f"API_INIT: Mem0 service not available: {e}")
            self.mem0_service = None
        
        # Use fixed demo user ID
        self.demo_user_id = DEMO_USER_ID
        
        # Initialize React scheduling agent
        self.chatbot = ReactSchedulingAgent(
            scheduler_service=self.scheduler_service,
            openai_api_key=openai_api_key,
            demo_user_id=self.demo_user_id
        )
        
        logger.info(f"API_INIT: System initialized with demo user: {self.demo_user_id}")
        logger.info(f"API_INIT: React scheduling agent ready")
        logger.info(f"API_INIT: All services loaded successfully")
    
    def _setup_demo_user(self) -> str:
        """Automatically set up demo user with default email and role"""
        try:
            # Check if demo user already exists
            existing_user = self.supabase.table("users").select("*").eq("email", DEFAULT_EMAIL).execute()
            
            if existing_user.data:
                user_id = existing_user.data[0]["id"]
                logger.info(f"DEMO_USER_SETUP: Using existing demo user: {DEFAULT_EMAIL}")
                return user_id
            else:
                # Create new demo user
                result = self.supabase.table("users").insert({
                    "email": DEFAULT_EMAIL,
                    "role": DEFAULT_ROLE.value,
                    "timezone": "UTC"
                }).execute()
                
                user_id = result.data[0]["id"]
                logger.info(f"DEMO_USER_SETUP: Created demo user: {DEFAULT_EMAIL} ({DEFAULT_ROLE.value})")
                return user_id
                
        except Exception as e:
            logger.warning(f"DEMO_USER_SETUP: Error setting up demo user: {e}")
            # Generate a fallback UUID for demo purposes
            return str(uuid.uuid4())
     
    def _get_type_icon(self, event_type: str) -> str:
        """Get text icon for event type (no emojis for better compatibility)"""
        icon_map = {
            'meeting': '[MEET]',
            'work': '[WORK]',
            'study': '[STUDY]',
            'exercise': '[FITNESS]',
            'health': '[HEALTH]',
            'personal': '[PERSONAL]',
            'social': '[SOCIAL]',
            'travel': '[TRAVEL]',
            'food': '[FOOD]',
            'shopping': '[SHOP]'
        }
        return icon_map.get(event_type.lower(), '[EVENT]')
    
    def _format_date_display(self, dt: datetime) -> str:
        """Format date for display: 'Wed, July 30'"""
        return dt.strftime("%a, %B %d")
    
    def _format_time_display(self, start_dt: datetime, end_dt: datetime) -> str:
        """Format time for display: '2:00pm - 2:30pm'"""
        # Use cross-platform format (Windows doesn't support %-I)
        start_time = start_dt.strftime("%I:%M%p").lower().lstrip('0')
        end_time = end_dt.strftime("%I:%M%p").lower().lstrip('0')
        return f"{start_time} - {end_time}"
    
    def _generate_priorities_array(self, current_priority: str) -> list:
        """Generate priorities array for chat event card display"""
        priorities = [
            {"label": "Low", "color": "low", "active": current_priority == "low"},
            {"label": "Med", "color": "medium", "active": current_priority == "medium"},
            {"label": "High", "color": "high", "active": current_priority == "high"}
        ]
        return priorities
    
    def _generate_availability_info(self) -> dict:
        """Generate availability information (mock data for demo)"""
        return {
            "label": "Available",
            "membersAvailable": "4/4 group members available",
            "conflicts": []  # Empty for demo
        }
    
    async def handle_onboarding(self, preferences: list, timestamp: str, user_id: str = None) -> dict:
        """Handle onboarding preferences and return OnboardingResponse"""
        try:
            # Always use default user regardless of provided user_id
            actual_user_id = self.demo_user_id
            logger.info(f"ONBOARDING: Using demo user ID: {actual_user_id}")
            
            # Process each preference like the prototype does
            task_types_created = 0
            preferences_applied = 0
            
            for i, preference in enumerate(preferences, 1):
                logger.info(f"ONBOARDING_STEP_{i}: Processing preference: '{preference}'")
                
                # Simple extraction - just use the preference text as task name
                task_name = preference.strip()
                
                if task_name:
                    logger.info(f"ONBOARDING_STEP_{i}: Extracted task name: '{task_name}'")
                    
                    try:
                        # Check for existing task type
                        existing_task_types = await self.task_type_service.get_user_task_types(actual_user_id)
                        exact_match = None
                        for existing_task in existing_task_types:
                            if existing_task.task_type.lower().strip() == task_name.lower().strip():
                                exact_match = existing_task
                                break
                        
                        if exact_match:
                            task_type = exact_match
                            logger.info(f"ONBOARDING_STEP_{i}: Found existing task type: {task_type.task_type}")
                            preferences_applied += 1
                        else:
                            # Create new task type
                            try:
                                task_type = await self.task_type_service.create_task_type(
                                    actual_user_id, task_name, 
                                    description=f"Created from onboarding: {preference}"
                                )
                                logger.info(f"ONBOARDING_STEP_{i}: Created new task type: {task_type.task_type}")
                                task_types_created += 1
                                preferences_applied += 1
                            except Exception as create_error:
                                if "duplicate key" in str(create_error).lower():
                                    logger.warning(f"ONBOARDING_STEP_{i}: Task type '{task_name}' already exists")
                                    preferences_applied += 1
                                else:
                                    logger.error(f"ONBOARDING_STEP_{i}: Failed to create task type: {create_error}")
                        
                    except Exception as e:
                        logger.warning(f"ONBOARDING_STEP_{i}: Could not process task type '{task_name}': {e}")
                else:
                    logger.warning(f"ONBOARDING_STEP_{i}: Could not extract task name from: '{preference}'")
            
            return {
                'success': True,
                'message': f'Processed {len(preferences)} preferences: {task_types_created} new task types, {preferences_applied} total applied',
                'userId': actual_user_id,
                'taskTypesCreated': task_types_created,
                'preferencesApplied': preferences_applied
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Onboarding error: {str(e)}',
                'userId': self.demo_user_id if hasattr(self, 'demo_user_id') else None
            }

# Initialize API instance
scheduler_api = SmartSchedulerAPI()

@app.route('/api/schedule', methods=['POST'])
def schedule():
    """Direct pattern-based scheduling endpoint"""
    try:
        data = request.get_json()
        task = data.get('task', '')
        
        logger.info(f"API /schedule: Request received - task: '{task}'")
        
        if not task:
            logger.warning("API /schedule: Missing task field")
            return jsonify(None), 400
        
        logger.info(f"DIRECT SCHEDULE REQUEST: '{task}'")
        logger.info(f"Using demo user ID: {scheduler_api.demo_user_id}")
        
        # Create new event loop for async operation
        logger.info("Creating new event loop for async operation")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Call scheduler_service.schedule_with_pattern directly with defaults
            logger.info("Calling scheduler_service.schedule_with_pattern")
            logger.info(f"Scheduler service available: {scheduler_api.scheduler_service is not None}")
            logger.info(f"Task type service available: {scheduler_api.task_type_service is not None}")
            logger.info(f"OpenAI client available: {scheduler_api.task_type_service.openai_client is not None}")
            
            result = loop.run_until_complete(
                scheduler_api.scheduler_service.schedule_with_pattern(
                    user_id=scheduler_api.demo_user_id,
                    summary=task,
                    openai_client=scheduler_api.task_type_service.openai_client
                )
            )
            
            logger.info(f"Raw result from schedule_with_pattern: {result}")
            logger.info(f"Result type: {type(result)}")
            
            if result:
                # Result is now just the event ID string
                event_id = result if isinstance(result, str) else None
                success = bool(event_id)
                logger.info(f"Extracted eventId: {event_id}")
                logger.info(f"Success flag: {success}")
                
                if event_id and success:
                    logger.info(f"API /schedule: Success - event ID: {event_id}")
                    return jsonify(event_id)
                else:
                    logger.warning(f"API /schedule: No event ID or failed - returning None")
                    return jsonify(None)
            else:
                logger.warning(f"API /schedule: No result returned - returning None")
                return jsonify(None)
                
        except Exception as async_error:
            logger.error(f"API /schedule ASYNC ERROR: {str(async_error)}")
            logger.error(f"API /schedule ASYNC ERROR TYPE: {type(async_error)}")
            import traceback
            logger.error(f"API /schedule ASYNC TRACEBACK: {traceback.format_exc()}")
            return jsonify(None), 500
        finally:
            loop.close()
            logger.info("Event loop closed")
        
    except Exception as e:
        logger.error(f"API /schedule ERROR: {str(e)}")
        logger.error(f"API /schedule ERROR TYPE: {type(e)}")
        import traceback
        logger.error(f"API /schedule TRACEBACK: {traceback.format_exc()}")
        return jsonify(None), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """LangGraph chatbot endpoint for conversational scheduling"""
    try:
        # Step 1: Parse and validate request
        logger.info("API_CHAT_STEP_1: Processing chat API request")
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', scheduler_api.demo_user_id)
        
        logger.info(f"API_CHAT_STEP_1: Request data parsed")
        logger.info(f"API_CHAT_STEP_1: Message: '{message}'")
        logger.info(f"API_CHAT_STEP_1: User ID: {user_id}")
        logger.info(f"API_CHAT_STEP_1: Message length: {len(message)} characters")
        
        if not message:
            logger.warning("API_CHAT_STEP_1: Validation failed - missing message field")
            return jsonify(None), 400
        
        # Step 2: Validate service availability
        logger.info("API_CHAT_STEP_2: Validating service availability")
        logger.info(f"API_CHAT_STEP_2: Chatbot service available: {scheduler_api.chatbot is not None}")
        logger.info(f"API_CHAT_STEP_2: Demo user ID: {scheduler_api.demo_user_id}")
        
        try:
            # Step 3: Invoke chatbot
            logger.info("API_CHAT_STEP_3: Invoking React scheduling agent")
            event_id = scheduler_api.chatbot.chat(message, user_id)
            
            logger.info(f"API_CHAT_STEP_3: Chatbot invocation completed")
            logger.info(f"API_CHAT_STEP_3: Raw result: {event_id}")
            logger.info(f"API_CHAT_STEP_3: Result type: {type(event_id)}")
            
            # Step 4: Process and return result
            logger.info("API_CHAT_STEP_4: Processing chatbot result")
            if event_id and event_id != "None":
                logger.info(f"API_CHAT_STEP_4: Success - returning event ID: {event_id}")
                return jsonify(event_id)
            else:
                logger.info(f"API_CHAT_STEP_4: No event scheduled - returning None")
                return jsonify(None)
                
        except Exception as chat_error:
            logger.error(f"API_CHAT_ERROR: Chatbot execution failed")
            logger.error(f"API_CHAT_ERROR: Error type: {type(chat_error).__name__}")
            logger.error(f"API_CHAT_ERROR: Error message: {str(chat_error)}")
            import traceback
            logger.error(f"API_CHAT_ERROR: Full traceback: {traceback.format_exc()}")
            return jsonify(None), 500
        
    except Exception as e:
        logger.error(f"API_CHAT_ERROR: Request processing failed")
        logger.error(f"API_CHAT_ERROR: Error type: {type(e).__name__}")
        logger.error(f"API_CHAT_ERROR: Error message: {str(e)}")
        import traceback
        logger.error(f"API_CHAT_ERROR: Full traceback: {traceback.format_exc()}")
        return jsonify(None), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({'status': 'healthy'})

@app.route('/api/onboarding', methods=['POST'])
def onboarding():
    """Handle onboarding preferences"""
    try:
        data = request.get_json()
        preferences = data.get('preferences', [])
        
        if not preferences:
            return jsonify({
                'success': False,
                'message': 'Missing preferences field'
            }), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            scheduler_api.handle_onboarding(preferences, datetime.now().isoformat())
        )
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get learned patterns"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        task_types = loop.run_until_complete(
            scheduler_api.task_type_service.get_user_task_types(scheduler_api.demo_user_id)
        )
        
        loop.close()
        
        patterns = []
        for task_type in task_types:
            patterns.append({
                'task_type': task_type.task_type,
                'completion_count': task_type.completion_count,
                'importance_score': task_type.importance_score
            })
        
        return jsonify({
            'success': True,
            'patterns': patterns,
            'total_task_types': len(task_types)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Smart Scheduler API")
    print("Endpoints:")
    print("  POST /api/schedule - Direct pattern-based scheduling")
    print("  POST /api/chat - LangGraph conversational scheduling")
    print("  POST /api/onboarding - Save preferences") 
    print("  GET  /api/status - View patterns")
    print("  GET  /health - Health check")
    print()
    print("Two scheduling approaches:")
    print("  /api/schedule - Direct API call to schedule_with_pattern")
    print("  /api/chat - Natural language chatbot with LangGraph")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000) 