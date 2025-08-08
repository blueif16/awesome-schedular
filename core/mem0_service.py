"""
Mem0 Service - Centralized Memory Management
Handles all mem0 operations for the smart scheduler using AsyncMemory
Simplified structure with 4 memory categories: priority, habit, energy, context
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from langchain_core.tools import tool
import sys
import os
import uuid

# Handle imports for both package and standalone execution
try:
    # Try package-relative import first
    from ..models import TaskType, ScheduleEventRequest
except ImportError:
    # Fallback for standalone execution - add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from models import TaskType, ScheduleEventRequest
# Handle config imports for standalone execution
try:
    from .config import get_openai_client, get_instructor_client
except ImportError:
    from config import get_openai_client, get_instructor_client

# Import MemoryCategorization from the local models file
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'models.py')
spec = importlib.util.spec_from_file_location("core_models", models_path)
core_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_models)
MemoryCategorization = core_models.MemoryCategorization
from dotenv import load_dotenv


load_dotenv()

MEM0_API_KEY = os.getenv("MEM0_API_KEY")



class SchedulerMemoryService:
    """Simplified Mem0 structure for smart scheduler with 4 categories"""
    
    def __init__(self):
        self.memory = None
        self.is_available = False
    
    async def initialize(self):
        """Initialize Memory service"""
        try:
            from mem0 import Memory
            self.memory = Memory(api_key=MEM0_API_KEY)
            self.is_available = True
            print("✅ SchedulerMemoryService initialized")
            return True
        except Exception as e:
            print(f"⚠️ Could not initialize Memory: {e}")
            return False
    
    # Category 1: Priority & Importance
    async def add_priority(self, user_id: str, task: str, signal: str):
        """Store task importance signals - fire and forget"""
        if not self.is_available:
            return False
        
        # Fire-and-forget: start task in background, return immediately
        asyncio.create_task(self._store_priority(user_id, task, signal))
        return True
    
    async def _store_priority(self, user_id: str, task: str, signal: str):
        """Internal method to actually store priority data"""
        try:
            await self.memory.add(
                f"{task}: {signal}",
                user_id=user_id,
                metadata={
                    "category": "priority",
                    "task": task,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to store priority memory: {e}")
    
    # Category 2: Habits & Patterns  
    async def add_habit(self, user_id: str, pattern: str):
        """Store behavioral patterns and preferences - fire and forget"""
        if not self.is_available:
            return False
        
        # Fire-and-forget: start task in background, return immediately
        asyncio.create_task(self._store_habit(user_id, pattern))
        return True
    
    async def _store_habit(self, user_id: str, pattern: str):
        """Internal method to actually store habit data"""
        try:
            await self.memory.add(
                pattern,
                user_id=user_id,
                metadata={
                    "category": "habit",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to store habit memory: {e}")
    
    # Category 3: Energy & Performance
    async def add_energy(self, user_id: str, observation: str):
        """Store energy patterns and performance observations - fire and forget"""
        if not self.is_available:
            return False
        
        # Fire-and-forget: start task in background, return immediately
        asyncio.create_task(self._store_energy(user_id, observation))
        return True
    
    async def _store_energy(self, user_id: str, observation: str):
        """Internal method to actually store energy data"""
        try:
            await self.memory.add(
                observation,
                user_id=user_id,
                metadata={
                    "category": "energy",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to store energy memory: {e}")
    
    # Category 4: Context & Goals
    async def add_context(self, user_id: str, context: str):
        """Store current goals, feelings, life context, details - fire and forget"""
        if not self.is_available:
            return False
        
        # Fire-and-forget: start task in background, return immediately
        asyncio.create_task(self._store_context(user_id, context))
        return True
    
    async def _store_context(self, user_id: str, context: str):
        """Internal method to actually store context data"""
        try:
            await self.memory.add(
                context,
                user_id=user_id,
                metadata={
                    "category": "context",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to store context memory: {e}")
    
    # Query methods for each category
    async def get_priority_context(self, user_id: str, task_type: str) -> List[Dict]:
        """Get importance/priority insights for a specific task"""
        if not self.is_available:
            return []
        results = await self.memory.search(
            query=f"{task_type} importance priority critical essential deadline",
            user_id=user_id,
            filters={"category": "priority"},
            limit=10
        )
        return results
    
    async def get_habit_patterns(self, user_id: str) -> List[Dict]:
        """Get user's behavioral patterns"""
        if not self.is_available:
            return []
        results = await self.memory.search(
            query="morning afternoon evening prefer time schedule pattern",
            user_id=user_id,
            filters={"category": "habit"},
            limit=10
        )
        return results
    
    async def get_energy_patterns(self, user_id: str) -> List[Dict]:
        """Get energy and performance patterns"""
        if not self.is_available:
            return []
        results = await self.memory.search(
            query="energy tired focused productive peak low performance",
            user_id=user_id,
            filters={"category": "energy"},
            limit=10
        )
        return results
    
    async def get_life_context(self, user_id: str) -> List[Dict]:
        """Get current goals, feelings, and context"""
        if not self.is_available:
            return []
        results = await self.memory.search(
            query="goal current feeling deadline project focus",
            user_id=user_id,
            filters={"category": "context"},
            limit=10
        )
        return results
    
    async def get_full_context_for_task_creation(self, user_id: str, task_type: str, description: Optional[str] = None) -> str:
        """Prepare comprehensive context for LLM task analysis"""
        if not self.is_available:
            return ""
        
        # Get all relevant memories
        priority_memories = await self.get_priority_context(user_id, task_type)
        habit_memories = await self.get_habit_patterns(user_id)
        energy_memories = await self.get_energy_patterns(user_id)
        context_memories = await self.get_life_context(user_id)
        
        # Format for LLM context
        context_parts = []
        
        if priority_memories:
            priority_text = "\n".join([m.get('memory', '') for m in priority_memories[:3]])
            context_parts.append(f"TASK IMPORTANCE CONTEXT:\n{priority_text}")
        
        if habit_memories:
            habit_text = "\n".join([m.get('memory', '') for m in habit_memories[:3]])
            context_parts.append(f"USER SCHEDULING HABITS:\n{habit_text}")
        
        if energy_memories:
            energy_text = "\n".join([m.get('memory', '') for m in energy_memories[:3]])
            context_parts.append(f"ENERGY PATTERNS:\n{energy_text}")
        
        if context_memories:
            context_text = "\n".join([m.get('memory', '') for m in context_memories[:3]])
            context_parts.append(f"CURRENT CONTEXT & GOALS:\n{context_text}")
        
        return "\n\n".join(context_parts) if context_parts else ""


class Mem0Service:
    def __init__(self, memory_service=None, task_type_service=None):
        """Initialize Mem0 service with optional AsyncMemory backend and task type service"""
        self.memory_service = memory_service
        self.scheduler_memory = SchedulerMemoryService()  # New simplified structure
        self.task_type_service = task_type_service
        self.is_available = memory_service is not None
    
    async def initialize_memory_service(self):
        """Initialize AsyncMemory service from mem0"""
        if self.memory_service:
            return True
        
        try:
            from mem0 import AsyncMemory
            self.memory_service = AsyncMemory()
            self.is_available = True
            print("✅ AsyncMemory service initialized successfully")
            
            # Also initialize simplified scheduler memory
            await self.scheduler_memory.initialize()
            return True
        except Exception as e:
            print(f"⚠️ Could not initialize AsyncMemory service: {e}")
            self.is_available = False
            # Try to initialize simplified memory anyway
            await self.scheduler_memory.initialize()
            return False
    
    async def query_scheduling_context(self, 
                                     user_id: str, 
                                     task_type: TaskType,
                                     request: ScheduleEventRequest) -> str:
        """Query both structured and unstructured memory for actionable task insights"""
        
        # Try simplified memory first
        context_parts = []
        if self.scheduler_memory.is_available:
            try:
                # Get structured context using 4 categories
                structured_context = await self.scheduler_memory.get_full_context_for_task_creation(
                    user_id, task_type.task_type, getattr(request, 'description', None)
                )
                if structured_context:
                    context_parts.append("STRUCTURED INSIGHTS:\n" + structured_context)
            except Exception as e:
                print(f"⚠️ Failed to get structured context: {e}")
        
        # Fallback to original AsyncMemory search
        if not self.is_available:
            if context_parts:
                return "\n\n".join(context_parts) + f"\n\nCURRENT TASK PROPERTIES: importance={task_type.importance_score:.2f}, cognitive_load={task_type.cognitive_load:.2f}"
            else:
                return f"Memory service not available. Current task properties: importance={task_type.importance_score:.2f}, cognitive_load={task_type.cognitive_load:.2f}"
            
        try:
            # Search for actionable task insights
            search_queries = [
                f"{task_type.task_type} importance cognitive load insights",
                f"task insights for {task_type.task_type}",
                f"{request.title} importance cognitive complexity",
                "task importance and cognitive load patterns"
            ]
            
            relevant_insights = []
            
            for query in search_queries:
                try:
                    results = await self.memory_service.search(
                        user_id=user_id,
                        query=query,
                        limit=5
                    )
                    
                    if results and len(results) > 0:
                        for result in results:
                            # Extract relevant information from memory
                            if hasattr(result, 'memory') and result.memory:
                                relevant_insights.append(result.memory)
                            elif hasattr(result, 'text') and result.text:
                                relevant_insights.append(result.text)
                except Exception as search_error:
                    print(f"⚠️ Mem0 search failed for query '{query}': {search_error}")
                    continue
            
            if relevant_insights:
                # Format actionable insights for LLM context
                unstructured_context = "LEARNED TASK INSIGHTS:\n"
                for insight in relevant_insights[:5]:  # Limit to 5 most relevant
                    unstructured_context += f"- {insight}\n"
                context_parts.append(unstructured_context)
            
            # Add current task type properties
            task_properties = f"CURRENT TASK TYPE PROPERTIES:\n"
            task_properties += f"- Importance Score: {task_type.importance_score:.2f} (0.0=low priority, 1.0=critical)\n"
            task_properties += f"- Cognitive Load: {task_type.cognitive_load:.2f} (0.0=easy/relaxing, 1.0=intense focus)\n"
            task_properties += f"- Total Completions: {task_type.completion_count}\n"
            context_parts.append(task_properties)
            
            return "\n\n".join(context_parts) if context_parts else f"Current task properties: importance={task_type.importance_score:.2f}, cognitive_load={task_type.cognitive_load:.2f}"
                
        except Exception as e:
            print(f"⚠️ Failed to query Mem0 for task insights: {e}")
            return "Memory service unavailable."
    
    @tool
    async def extract_and_store_text_insights(self, 
                                         user_id: str, 
                                         text_content: str,
                                         context: str = "conversation") -> str:
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
        
        # Add logging imports at function level to avoid circular imports
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🧠 MEM0 TOOL INVOKED")
        logger.info(f"📝 MEM0 PARAMS: user_id={user_id}, context='{context}'")
        logger.info(f"📝 MEM0 PARAMS: text_content='{text_content}'")
        
        # Get global instructor client for structured outputs
        try:
            instructor_client = get_instructor_client()
            logger.info("✅ Instructor client obtained successfully")
        except Exception as e:
            logger.error(f"❌ Failed to get instructor client: {e}")
            return f"Error: Could not initialize LLM client - {str(e)}"
        
        prompt = f"""
Analyze this user content and extract insights for scheduling optimization:

USER CONTENT: "{text_content}"
CONTEXT: "{context}"

Categorize into 4 memory types:

1. PRIORITY: Importance signals, urgency, deadlines, critical tasks
   - Look for: "important", "urgent", "deadline", "critical", "must do", "priority"
   
2. HABIT: Time preferences, routines, scheduling patterns  
   - Look for: "morning", "evening", "prefer", "usually", "schedule", "routine"
   
3. ENERGY: Focus, productivity, energy patterns, performance
   - Look for: "focused", "tired", "energy", "productive", "peak", "performance"
   
4. CONTEXT: Goals, feelings, life situation, general information
   - Everything else that provides useful scheduling context

Extract specific, actionable insights for each category. Return empty arrays for categories with no relevant content.
"""
        
        try:
            # Use instructor for structured output
            result = instructor_client.chat.completions.create(
                model="gpt-4o",
                response_model=MemoryCategorization,
                messages=[
                    {"role": "system", "content": "You are a scheduling assistant that categorizes user content into memory types for intelligent task scheduling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_retries=3
            )
            
            # Access Pydantic model attributes directly
            confidence = result.confidence
            if confidence < 0.3:
                return f"⚠️ Low confidence ({confidence:.2f}) in content categorization - no insights stored"
            
            # Store insights in appropriate categories using fire-and-forget
            stored_categories = []
            
            if self.scheduler_memory.is_available:
                # Store priority insights - fire and forget
                for insight in result.priority_insights:
                    if insight.strip():
                        self.scheduler_memory.add_priority(user_id, context or "general", insight)
                        if "priority" not in stored_categories:
                            stored_categories.append("priority")
                
                # Store habit insights - fire and forget
                for insight in result.habit_insights:
                    if insight.strip():
                        self.scheduler_memory.add_habit(user_id, insight)
                        if "habit" not in stored_categories:
                            stored_categories.append("habit")
                
                # Store energy insights - fire and forget
                for insight in result.energy_insights:
                    if insight.strip():
                        self.scheduler_memory.add_energy(user_id, insight)
                        if "energy" not in stored_categories:
                            stored_categories.append("energy")
                
                # Store context insights - fire and forget
                for insight in result.context_insights:
                    if insight.strip():
                        self.scheduler_memory.add_context(user_id, insight)
                        if "context" not in stored_categories:
                            stored_categories.append("context")
                
                total_insights = len(result.priority_insights) + len(result.habit_insights) + len(result.energy_insights) + len(result.context_insights)
                
                if stored_categories:
                    return f"📝 Stored {total_insights} insights in {len(stored_categories)} categories: {', '.join(stored_categories)} (confidence: {confidence:.2f})"
                else:
                    return "No actionable insights found to store"
            else:
                return "❌ Error: Memory service not available"
                
        except Exception as e:
            return f"❌ Error in LLM content categorization: {str(e)}"
    
    # Removed redundant _extract_time_preferences function
    # Time preference extraction is now handled by task_type_service._analyze_task_patterns_with_llm
    
    async def store_onboarding_preferences(self, 
                                         user_id: str,
                                         user_input: str,
                                         context: str = "onboarding",
                                         openai_client=None) -> Dict[str, bool]:
        """Store onboarding preferences using LLM categorization"""
        
        # Use the new LLM-powered categorization
        return await self.extract_and_store_text_insights(
            user_id=user_id,
            text_content=user_input,
            context=context,
            openai_client=openai_client
        )
    
    async def search_user_patterns(self, 
                                 user_id: str, 
                                 query: str, 
                                 limit: int = 5) -> List[str]:
        """Search for user patterns in memory"""
        if not self.is_available:
            return []
            
        try:
            results = await self.memory_service.search(
                user_id=user_id,
                query=query,
                limit=limit
            )
            
            patterns = []
            if results:
                for result in results:
                    if hasattr(result, 'memory') and result.memory:
                        patterns.append(result.memory)
                    elif hasattr(result, 'text') and result.text:
                        patterns.append(result.text)
            
            return patterns
            
        except Exception as e:
            print(f"⚠️ Error searching user patterns: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get mem0 service status"""
        return {
            "available": self.is_available,
            "service_type": "mem0" if self.memory_service else "none",
            "initialized": self.memory_service is not None,
            "scheduler_memory_available": self.scheduler_memory.is_available
        }


# Global instance for easy access
_mem0_service_instance = None

def get_mem0_service() -> Mem0Service:
    """Get the global mem0 service instance"""
    global _mem0_service_instance
    if _mem0_service_instance is None:
        _mem0_service_instance = Mem0Service()
    return _mem0_service_instance

def set_task_type_service(task_type_service):
    """Set the task type service for the global mem0 service instance"""
    service = get_mem0_service()
    service.task_type_service = task_type_service

def set_scheduler_memory_service(scheduler_memory_service):
    """Set the scheduler memory service for the global mem0 service instance"""
    service = get_mem0_service()
    service.scheduler_memory = scheduler_memory_service

async def initialize_mem0_service(memory_service=None, task_type_service=None) -> Mem0Service:
    """Initialize and return mem0 service"""
    service = get_mem0_service()
    if memory_service:
        service.memory_service = memory_service
        service.is_available = True
    else:
        await service.initialize_memory_service()
    
    if task_type_service:
        service.task_type_service = task_type_service
        
    return service

async def get_scheduler_memory() -> SchedulerMemoryService:
    """Get or create simplified scheduler memory service"""
    memory_service = SchedulerMemoryService()
    await memory_service.initialize()
    return memory_service 