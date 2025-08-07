"""
Mem0 Service - Centralized Memory Management
Handles all mem0 operations for the smart scheduler using AsyncMemory
Simplified structure with 4 memory categories: priority, habit, energy, context
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from models import TaskType, ScheduleEventRequest
import uuid


class SchedulerMemoryService:
    """Simplified Mem0 structure for smart scheduler with 4 categories"""
    
    def __init__(self):
        self.memory = None
        self.is_available = False
    
    async def initialize(self):
        """Initialize Memory service"""
        try:
            from mem0 import Memory
            self.memory = Memory()
            self.is_available = True
            print("✅ SchedulerMemoryService initialized")
            return True
        except Exception as e:
            print(f"⚠️ Could not initialize Memory: {e}")
            return False
    
    # Category 1: Priority & Importance
    async def add_priority(self, user_id: str, task: str, signal: str):
        """Store task importance signals"""
        if not self.is_available:
            return False
        await self.memory.add(
            f"{task}: {signal}",
            user_id=user_id,
            metadata={
                "category": "priority",
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    # Category 2: Habits & Patterns  
    async def add_habit(self, user_id: str, pattern: str):
        """Store behavioral patterns and preferences"""
        if not self.is_available:
            return False
        await self.memory.add(
            pattern,
            user_id=user_id,
            metadata={
                "category": "habit",
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    # Category 3: Energy & Performance
    async def add_energy(self, user_id: str, observation: str):
        """Store energy patterns and performance observations"""
        if not self.is_available:
            return False
        await self.memory.add(
            observation,
            user_id=user_id,
            metadata={
                "category": "energy",
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    # Category 4: Context & Goals
    async def add_context(self, user_id: str, context: str):
        """Store current goals, feelings, life context, details"""
        if not self.is_available:
            return False
        await self.memory.add(
            context,
            user_id=user_id,
            metadata={
                "category": "context",
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
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
    
    async def extract_and_store_text_insights(self, 
                                         user_id: str, 
                                         text_content: str,
                                         context: str = "",
                                         openai_client=None) -> Dict[str, bool]:
        """LLM-powered extraction and categorization of user content into 4 memory categories"""
        
        if not openai_client:
            print("⚠️ OpenAI client required for LLM categorization")
            return {"priority": False, "habit": False, "energy": False, "context": False}
        
        function_schema = {
            "name": "categorize_user_content",
            "description": "Categorize user content into memory categories and extract actionable insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "priority_insights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task importance signals, urgency markers, deadlines, critical needs"
                    },
                    "habit_insights": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Time preferences, routine patterns, scheduling habits, recurring behaviors"
                    },
                    "energy_insights": {
                        "type": "array",
                        "items": {"type": "string"}, 
                        "description": "Energy levels, focus patterns, productivity observations, performance notes"
                    },
                    "context_insights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Goals, feelings, life situation, general context, other relevant information"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence in categorization (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["priority_insights", "habit_insights", "energy_insights", "context_insights", "confidence"]
            }
        }
        
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
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a scheduling assistant that categorizes user content into memory types for intelligent task scheduling."},
                    {"role": "user", "content": prompt}
                ],
                functions=[function_schema],
                function_call={"name": "categorize_user_content"},
                temperature=0.2
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "categorize_user_content":
                result = json.loads(function_call.arguments)
                
                confidence = result.get("confidence", 0.0)
                if confidence < 0.3:
                    print(f"⚠️ Low confidence ({confidence:.2f}) in content categorization")
                    return {"priority": False, "habit": False, "energy": False, "context": False}
                
                # Store insights in appropriate categories
                storage_results = {"priority": False, "habit": False, "energy": False, "context": False}
                
                if self.scheduler_memory.is_available:
                    # Store priority insights
                    priority_insights = result.get("priority_insights", [])
                    for insight in priority_insights:
                        if insight.strip():
                            await self.scheduler_memory.add_priority(user_id, context or "general", insight)
                            storage_results["priority"] = True
                    
                    # Store habit insights  
                    habit_insights = result.get("habit_insights", [])
                    for insight in habit_insights:
                        if insight.strip():
                            await self.scheduler_memory.add_habit(user_id, insight)
                            storage_results["habit"] = True
                    
                    # Store energy insights
                    energy_insights = result.get("energy_insights", [])
                    for insight in energy_insights:
                        if insight.strip():
                            await self.scheduler_memory.add_energy(user_id, insight)
                            storage_results["energy"] = True
                    
                    # Store context insights
                    context_insights = result.get("context_insights", [])
                    for insight in context_insights:
                        if insight.strip():
                            await self.scheduler_memory.add_context(user_id, insight)
                            storage_results["context"] = True
                    
                    total_insights = len(priority_insights) + len(habit_insights) + len(energy_insights) + len(context_insights)
                    stored_categories = sum(storage_results.values())
                    print(f"✅ LLM extracted {total_insights} insights → stored in {stored_categories} categories (confidence: {confidence:.2f})")
                
                return storage_results
            else:
                print("⚠️ LLM did not return expected categorization")
                return {"priority": False, "habit": False, "energy": False, "context": False}
                
        except Exception as e:
            print(f"❌ Error in LLM content categorization: {e}")
            return {"priority": False, "habit": False, "energy": False, "context": False}
    
    async def _extract_time_preferences(self, 
                                      user_id: str, 
                                      user_input: str,
                                      task_type: TaskType,
                                      openai_client=None,
                                      save_to_db: bool = True) -> None:
        """Extract specific time preferences using LLM with compact string format"""
        if not self.is_available or not openai_client:
            return
            
        try:
            function_schema = {
                "name": "extract_task_insights",
                "description": "Extract task importance, cognitive load, and time preferences from user input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_patterns": {
                            "type": "string",
                            "description": "Compact time patterns: 'days:hour_start-hour_end:boost,days:hour_start-hour_end:boost'. Days: 0-6=Mon-Sun, 0-4=weekdays, 5-6=weekend. Example: '0-6:6-11:0.8,5-6:17-21:0.9'"
                        },
                        "importance_score": {
                            "type": "number",
                            "description": "How important this task is to the user (0.0-1.0). Based on language cues like 'critical', 'urgent', 'love', 'passionate about', 'hate', 'must do'",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "cognitive_load": {
                            "type": "number", 
                            "description": "Mental difficulty/focus required (0.0-1.0). 0.0=easy/relaxing, 1.0=intense focus. Based on words like 'deep work', 'focus', 'thinking', 'relax', 'easy'",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in extraction (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["time_patterns", "importance_score", "cognitive_load", "confidence"]
                }
            }
            
            context = f"""
Extract actionable task insights from user input:

User Input: "{user_input}"
Task Type: "{task_type.task_type}"

ANALYZE FOR:

1. TIME PATTERNS (when they prefer this task):
   - "morning" → "0-6:6-11:0.8"
   - "not morning person" → "0-6:6-11:0.2" 
   - "evening focus" → "0-6:18-22:0.9"

2. IMPORTANCE SCORE (how much they care about this task):
   - High (0.8-1.0): "love", "passionate", "critical", "must do", "priority"
   - Medium (0.4-0.7): "like", "important", "should do"
   - Low (0.0-0.3): "hate", "avoid", "don't care", "if I have to"

3. COGNITIVE LOAD (mental effort required):
   - High (0.8-1.0): "deep work", "focus", "thinking", "complex", "demanding"
   - Medium (0.4-0.7): "moderate", "some focus", "planning"
   - Low (0.0-0.3): "relax", "easy", "mindless", "automatic", "simple"

Extract meaningful insights that will improve scheduling decisions.
"""
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract specific time and scheduling preferences from user statements. Be precise with time patterns and categorization."},
                    {"role": "user", "content": context}
                ],
                functions=[function_schema],
                function_call={"name": "extract_task_insights"},
                temperature=0.2
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "extract_task_insights":
                result = json.loads(function_call.arguments)
                
                time_patterns = result.get("time_patterns", "")
                importance_score = result.get("importance_score", 0.0)
                cognitive_load = result.get("cognitive_load", 0.0)
                confidence = result.get("confidence", 0.0)
                
                if confidence < 0.3:
                    print(f"⚠️ Low confidence ({confidence:.2f}) in preference extraction")
                    return
                
                # Store actionable task insights in mem0
                insights_text = f"Task '{task_type.task_type}' insights: importance={importance_score:.2f}, cognitive_load={cognitive_load:.2f}, time_patterns='{time_patterns}'"
                
                await self.memory_service.add(
                    user_id=user_id,
                    messages=[{
                        "role": "assistant",
                        "content": insights_text
                    }],
                    metadata={
                        "category": "task_insights",
                        "task_type": task_type.task_type,
                        "importance_score": importance_score,
                        "cognitive_load": cognitive_load,
                        "time_patterns": time_patterns,
                        "confidence": confidence,
                        "extracted_from": user_input,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Update the actual task type with meaningful properties
                updated_task_type = await self._update_task_type_insights(
                    task_type, importance_score, cognitive_load, confidence
                )
                
                # Persist changes to database if updates were made
                if confidence >= 0.5 and save_to_db:
                    await self._save_task_type_insights(updated_task_type)
                
                print(f"🎯 LLM extracted importance_score: {importance_score:.2f}, cognitive_load: {cognitive_load:.2f}")
                print(f"📋 Time patterns: '{time_patterns}' (confidence: {confidence:.2f})")
                print(f"✅ Updated task type with actionable insights")
                    
        except Exception as e:
            print(f"⚠️ Failed to extract time preferences with LLM: {e}")
    
    async def _update_task_type_insights(self, 
                                       task_type: TaskType, 
                                       importance_score: float,
                                       cognitive_load: float,
                                       confidence: float) -> TaskType:
        """Update task type with extracted insights if confidence is high enough"""
        
        if confidence < 0.5:
            print(f"⚠️ Confidence too low ({confidence:.2f}) to update task type properties")
            return task_type
        
        # Update importance_score (weighted average with existing if it exists)
        if task_type.importance_score > 0:
            # Blend with existing score
            task_type.importance_score = (task_type.importance_score * 0.7) + (importance_score * 0.3)
        else:
            task_type.importance_score = importance_score
        
        # Update cognitive_load (weighted average with existing)
        if task_type.cognitive_load != 0.5:  # 0.5 is default, so check if it's been set
            # Blend with existing score
            task_type.cognitive_load = (task_type.cognitive_load * 0.7) + (cognitive_load * 0.3)
        else:
            task_type.cognitive_load = cognitive_load
        
        print(f"🔄 Updated task type: importance={task_type.importance_score:.2f}, cognitive_load={task_type.cognitive_load:.2f}")
        return task_type
    
    async def _save_task_type_insights(self, task_type: TaskType) -> bool:
        """Save updated task type insights to database"""
        try:
            if not self.task_type_service:
                print(f"⚠️ Task type service not available - insights not persisted to database")
                return False
            
            # Update the task type in the database
            result = self.task_type_service.supabase.table("task_types").update({
                "importance_score": task_type.importance_score,
                "cognitive_load": task_type.cognitive_load,
                "updated_at": datetime.now().isoformat()
            }).eq("id", str(task_type.id)).execute()
            
            if result.data:
                print(f"💾 Saved task type insights to database: importance={task_type.importance_score:.2f}, cognitive_load={task_type.cognitive_load:.2f}")
                return True
            else:
                print(f"⚠️ Failed to save task type insights - no data returned")
                return False
                
        except Exception as e:
            print(f"❌ Error saving task type insights: {e}")
            return False
    
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