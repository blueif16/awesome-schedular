"""
Global Configuration and Shared Services
Provides centralized OpenAI client and service instances
"""

import os
import instructor
from langchain_openai import ChatOpenAI
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global OpenAI Client
def get_openai_client() -> ChatOpenAI:
    """Get the global OpenAI client instance"""
    global _openai_client
    if '_openai_client' not in globals():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        _openai_client = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.3
        )
    return _openai_client

# Global Instructor Client for Structured Outputs
def get_instructor_client():
    """Get the global Instructor client instance"""
    global _instructor_client
    if '_instructor_client' not in globals():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        _instructor_client = instructor.from_provider("openai/gpt-4o", api_key=api_key)
    return _instructor_client

# Global Supabase Client
def get_supabase_client():
    """Get the global Supabase client instance"""
    global _supabase_client
    if '_supabase_client' not in globals():
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not configured")
        
        _supabase_client = create_client(supabase_url, supabase_key)
    return _supabase_client

# Service Instances - Created once and reused
_task_type_service = None
_scheduler_service = None
_mem0_service = None

def get_task_type_service():
    """Get the global TaskTypeService instance"""
    global _task_type_service
    if _task_type_service is None:
        try:
            from .task_type_service import TaskTypeService
        except ImportError:
            # Fallback for direct script execution
            from task_type_service import TaskTypeService
        _task_type_service = TaskTypeService(get_supabase_client())
    return _task_type_service

def get_scheduler_service():
    """Get the global SchedulerService instance"""
    global _scheduler_service
    if _scheduler_service is None:
        try:
            from .scheduler_service import SchedulerService
        except ImportError:
            # Fallback for direct script execution
            from scheduler_service import SchedulerService
        _scheduler_service = SchedulerService(get_task_type_service())
    return _scheduler_service

def get_mem0_service():
    """Get the global Mem0Service instance"""
    global _mem0_service
    if _mem0_service is None:
        try:
            from .mem0_service import Mem0Service
        except ImportError:
            # Fallback for direct script execution
            from mem0_service import Mem0Service
        _mem0_service = Mem0Service()
    return _mem0_service 