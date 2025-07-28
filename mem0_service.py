"""
Mem0 Service - Centralized Memory Management
Handles all mem0 operations for the smart scheduler using AsyncMemory
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from models import TaskType, ScheduleEventRequest
import uuid


class Mem0Service:
    def __init__(self, memory_service=None, task_type_service=None):
        """Initialize Mem0 service with optional AsyncMemory backend and task type service"""
        self.memory_service = memory_service
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
            return True
        except Exception as e:
            print(f"⚠️ Could not initialize AsyncMemory service: {e}")
            self.is_available = False
            return False
    
    async def query_scheduling_context(self, 
                                     user_id: str, 
                                     task_type: TaskType,
                                     request: ScheduleEventRequest) -> str:
        """Query Mem0 for actionable task insights - importance, cognitive load, and time preferences"""
        if not self.is_available:
            return "Memory service not available."
            
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
                context_summary = "LEARNED TASK INSIGHTS:\n"
                for insight in relevant_insights[:5]:  # Limit to 5 most relevant
                    context_summary += f"- {insight}\n"
                
                # Add current task type properties
                context_summary += f"\nCURRENT TASK TYPE PROPERTIES:\n"
                context_summary += f"- Importance Score: {task_type.importance_score:.2f} (0.0=low priority, 1.0=critical)\n"
                context_summary += f"- Cognitive Load: {task_type.cognitive_load:.2f} (0.0=easy/relaxing, 1.0=intense focus)\n"
                context_summary += f"- Total Completions: {task_type.completion_count}\n"
                
                return context_summary
            else:
                # Return default task properties when no mem0 insights found
                return f"No previous insights found. Current task properties: importance={task_type.importance_score:.2f}, cognitive_load={task_type.cognitive_load:.2f}"
                
        except Exception as e:
            print(f"⚠️ Failed to query Mem0 for task insights: {e}")
            return "Memory service unavailable."
    
    async def store_scheduling_preferences(self, 
                                         user_id: str, 
                                         user_preferences: str,
                                         task_type: TaskType,
                                         detected_patterns: List[str],
                                         openai_client=None) -> bool:
        """Extract and store scheduling preferences in AsyncMemory"""
        if not self.is_available:
            print(f"⚠️ AsyncMemory not available - preferences stored locally only")
            return False
            
        try:
            # Store the full user input with context
            await self.memory_service.add(
                user_id=user_id,
                messages=[{
                    "role": "user",
                    "content": f"When scheduling {task_type.task_type}: {user_preferences}"
                }],
                metadata={
                    "category": "scheduling_preference",
                    "task_type": task_type.task_type,
                    "preference_type": "user_stated_preference",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Store each detected pattern separately for better retrieval
            for pattern in detected_patterns:
                await self.memory_service.add(
                    user_id=user_id,
                    messages=[{
                        "role": "assistant",
                        "content": f"User scheduling pattern: {pattern}"
                    }],
                    metadata={
                        "category": "scheduling_pattern",
                        "pattern_type": "extracted_preference",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            print(f"✅ Stored {len(detected_patterns) + 1} scheduling preferences in Mem0")
            
            # Also extract and store specific time-related preferences using LLM
            await self._extract_time_preferences(user_id, user_preferences, task_type, openai_client, save_to_db=True)
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to extract and store preferences in Mem0: {e}")
            return False
    
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
                                         task_name: str = "",
                                         preferences: List[str] = None,
                                         openai_client=None) -> bool:
        """Store onboarding preferences in AsyncMemory"""
        if not self.is_available:
            print(f"⚠️ Memory service not available - stored locally only")
            return False
            
        try:
            await self.memory_service.add(
                user_id=user_id,
                messages=[{
                    "role": "user",
                    "content": user_input
                }],
                metadata={
                    "category": "onboarding_preference",
                    "task_type": task_name if task_name else "general",
                    "preferences": preferences or [],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Also store extracted patterns
            if preferences:
                for preference in preferences:
                    await self.memory_service.add(
                        user_id=user_id,
                        messages=[{
                            "role": "assistant",
                            "content": f"User scheduling pattern: {preference}"
                        }],
                        metadata={
                            "category": "scheduling_pattern",
                            "pattern_type": "onboarding_extracted",
                            "task_type": task_name if task_name else "general",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
            # If task_name provided and openai_client available, also extract time preferences with LLM
            if task_name and openai_client:
                from models import TaskType
                # Create a mock task type for preference extraction (DO NOT SAVE TO DB)
                mock_task_type = TaskType(
                    id=uuid.uuid4(),
                    user_id=uuid.UUID(user_id),
                    task_type=task_name,
                    description="Onboarding task type for preference extraction",
                    weekly_habit_scores=[0.5] * 168,
                    slot_confidence=[[0.1 for _ in range(24)] for _ in range(7)],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Extract preferences but don't save mock task type to database
                await self._extract_time_preferences(user_id, user_input, mock_task_type, openai_client, save_to_db=False)
            
            print(f"✅ Stored {len(preferences or []) + 1} preference items in memory!")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not store in memory: {e}")
            return False
    
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
            "initialized": self.memory_service is not None
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