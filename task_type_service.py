"""
Task Type Service - Tier 2 Implementation
Handles task type creation, vector search, and similarity matching
"""

import openai
import uuid
import json
from typing import List, Optional, Dict, Any
from supabase import Client
from models import TaskType, TaskTypeSimilarity, initialize_neutral_weekly_habit_array, initialize_slot_confidence
from datetime import datetime


class TaskTypeService:
    def __init__(self, supabase: Client, openai_api_key: str):
        self.supabase = supabase
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
    
    def _parse_embedding(self, embedding_data: Any) -> Optional[List[float]]:
        """Parse embedding data from database - handles both string and list formats"""
        if embedding_data is None:
            return None
        
        # If already a list, return as-is
        if isinstance(embedding_data, list):
            return embedding_data
        
        # If string, try to parse as JSON
        if isinstance(embedding_data, str):
            try:
                parsed = json.loads(embedding_data)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not parse embedding string: {embedding_data[:50]}...")
                return None
        
        print(f"Warning: Unexpected embedding type: {type(embedding_data)}")
        return None
    
    def generate_embedding(self, task_type: str, description: str = None) -> List[float]:
        """Generate OpenAI embedding from task type and optional description"""
        try:
            # Combine task type and description for richer embedding
            text_parts = [task_type]
            if description and description.strip():
                text_parts.append(description.strip())
            
            combined_text = " - ".join(text_parts)
            
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=combined_text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    

    
    async def find_similar_task_type(self, user_id: str, 
                                   task_type: str,
                                   description: str = None,
                                   threshold: float = 0.4) -> Optional[TaskTypeSimilarity]:
        """Find similar task type using vector search"""
        
        # First, let's see what task types this user already has
        try:
            existing_result = self.supabase.table("task_types")\
                .select("task_type, completion_count")\
                .eq("user_id", user_id)\
                .execute()
            
            if existing_result.data:
                print(f"📊 USER HAS {len(existing_result.data)} EXISTING TASK TYPES:")
                for task in existing_result.data:
                    print(f"   • '{task['task_type']}' (completions: {task['completion_count']})")
            else:
                print(f"📊 USER HAS NO EXISTING TASK TYPES - this will be the first one!")
        except Exception as e:
            print(f"⚠️ Could not check existing task types: {e}")
        
        # Create search query for logging
        search_query = f"'{task_type}'"
        if description:
            search_query += f" + '{description}'"
        
        print(f"🔍 RAG Search: {search_query} (threshold: {threshold:.2f})")
        
        embedding = self.generate_embedding(task_type, description)
        
        try:
            # First, search with a low threshold to get the highest match regardless
            result_all = self.supabase.rpc(
                'match_task_types',
                {
                    'query_embedding': embedding,
                    'match_threshold': 0.0,  # Get all matches to find highest
                    'match_count': 1,
                    'target_user_id': user_id
                }
            ).execute()
            
            if result_all.data:
                best_match = result_all.data[0]
                best_similarity = best_match['similarity']
                best_task_name = best_match['task_type']
                
                # Always show the highest match found
                print(f"🎯 RAG Best Match: '{best_task_name}' (similarity: {best_similarity:.3f})")
                
                # Then decide whether to use it based on the threshold
                if best_similarity >= threshold:
                    print(f"✅ Similarity {best_similarity:.3f} >= {threshold:.2f} threshold - USING EXISTING TASK TYPE")
                else:
                    print(f"🔸 Similarity {best_similarity:.3f} < {threshold:.2f} threshold - will create new task type")
                
                # Convert to TaskType model
                task_type_obj = TaskType(
                    id=best_match['id'],
                    user_id=uuid.UUID(user_id),
                    task_type=best_match['task_type'],
                    description=best_match.get('description'),
                    weekly_habit_scores=best_match['weekly_habit_scores'],
                    slot_confidence=best_match['slot_confidence'],
                    completion_count=best_match['completion_count'],
                    completions_since_last_update=best_match['completions_since_last_update'],
                    typical_duration=best_match['typical_duration'],
                    importance_score=best_match['importance_score'],
                    recovery_hours=best_match['recovery_hours'],
                    cognitive_load=best_match.get('cognitive_load', 0.5),  # Default if missing
                    embedding=self._parse_embedding(best_match.get('embedding')),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Only return the match if it meets the threshold
                if best_similarity >= threshold:
                    return TaskTypeSimilarity(
                        task_type=task_type_obj,
                        similarity=best_match['similarity']
                    )
                else:
                    # Return None but we've already logged the best match found
                    return None
            else:
                print(f"❌ RAG Search: No task types found for user {user_id}")
                
        except Exception as e:
            print(f"❌ RAG Search Error: {e}")
            
        return None
    
    async def create_task_type(self, user_id: str, 
                             task_type: str, 
                             description: Optional[str] = None,
                             scheduler_service = None) -> TaskType:
        """Create new task type with LLM-analyzed cognitive load and importance"""
        
        print(f"🆕 Creating new task type: '{task_type}' for user {user_id[:8]}...")
        if description:
            print(f"   📝 Description: {description}")
        
        # Generate embedding from task type and description
        embedding = self.generate_embedding(task_type, description)
        print(f"   🧮 Generated embedding (dimensions: {len(embedding)})")
        
        # Analyze task characteristics using LLM (from scheduler service)
        if scheduler_service:
            task_analysis = await scheduler_service.analyze_task_characteristics(
                self.openai_client, task_type, description
            )
            print(f"   🧠 LLM Analysis - Cognitive Load: {task_analysis['cognitive_load']:.2f}, "
                  f"Importance: {task_analysis['importance_score']:.2f}, "
                  f"Duration: {task_analysis['typical_duration']:.1f}h")
        else:
            # Fallback to simple defaults if no scheduler service provided
            task_analysis = {
                "cognitive_load": 0.5,
                "importance_score": 0.5,
                "typical_duration": 1.0,
                "recovery_hours": 0.5
            }
            print(f"   🔄 Using default analysis (no scheduler service provided)")
        
        # Initialize neutral patterns and confidence matrix
        weekly_habit_scores = initialize_neutral_weekly_habit_array()
        slot_confidence = initialize_slot_confidence()
        print(f"   📊 Initialized neutral patterns and confidence matrix")
        
        new_task_type = {
            "user_id": user_id,
            "task_type": task_type,
            "description": description,
            "weekly_habit_scores": weekly_habit_scores,
            "slot_confidence": slot_confidence,
            "completion_count": 0,
            "completions_since_last_update": 0,
            "typical_duration": task_analysis['typical_duration'],
            "importance_score": task_analysis['importance_score'],
            "recovery_hours": task_analysis['recovery_hours'],
            "cognitive_load": task_analysis['cognitive_load'],
            "embedding": embedding
        }
        
        try:
            print(f"   💾 Inserting task type into database...")
            result = self.supabase.table("task_types").insert(new_task_type).execute()
            data = result.data[0]
            
            print(f"✅ Task type created successfully!")
            print(f"   🆔 ID: {data['id']}")
            print(f"   📊 Completion count: {data['completion_count']}")
            print(f"   ⭐ Importance score: {data['importance_score']}")
            
            return TaskType(
                id=data['id'],
                user_id=uuid.UUID(user_id),
                task_type=data['task_type'],
                description=data.get('description'),
                weekly_habit_scores=data['weekly_habit_scores'],
                slot_confidence=data['slot_confidence'],
                completion_count=data['completion_count'],
                completions_since_last_update=data['completions_since_last_update'],
                typical_duration=data['typical_duration'],
                importance_score=data['importance_score'],
                recovery_hours=data['recovery_hours'],
                cognitive_load=data.get('cognitive_load', 0.5),  # Default if missing
                embedding=self._parse_embedding(data.get('embedding')),
                created_at=data['created_at'],
                updated_at=data['updated_at']
            )
            
        except Exception as e:
            print(f"❌ Error creating task type '{task_type}': {e}")
            raise
    
    async def get_task_type(self, task_type_id: str) -> Optional[TaskType]:
        """Get task type by ID"""
        try:
            result = self.supabase.table("task_types")\
                .select("*")\
                .eq("id", task_type_id)\
                .execute()
            
            if result.data:
                data = result.data[0]
                return TaskType(
                    id=data['id'],
                    user_id=data['user_id'],
                    task_type=data['task_type'],
                    description=data.get('description'),
                    weekly_habit_scores=data['weekly_habit_scores'],
                    slot_confidence=data['slot_confidence'],
                    completion_count=data['completion_count'],
                    completions_since_last_update=data['completions_since_last_update'],
                    typical_duration=data['typical_duration'],
                    importance_score=data['importance_score'],
                    recovery_hours=data['recovery_hours'],
                    cognitive_load=data.get('cognitive_load', 0.5),  # Default if missing
                    embedding=self._parse_embedding(data.get('embedding')),
                    created_at=data['created_at'],
                    updated_at=data['updated_at']
                )
        except Exception as e:
            print(f"Error getting task type: {e}")
            
        return None
    
    async def get_user_task_types(self, user_id: str) -> List[TaskType]:
        """Get all task types for a user"""
        try:
            result = self.supabase.table("task_types")\
                .select("*")\
                .eq("user_id", user_id)\
                .execute()
            
            task_types = []
            for data in result.data:
                task_types.append(TaskType(
                    id=data['id'],
                    user_id=data['user_id'],
                    task_type=data['task_type'],
                    description=data.get('description'),
                    weekly_habit_scores=data['weekly_habit_scores'],
                    slot_confidence=data['slot_confidence'],
                    completion_count=data['completion_count'],
                    completions_since_last_update=data['completions_since_last_update'],
                    typical_duration=data['typical_duration'],
                    importance_score=data['importance_score'],
                    recovery_hours=data['recovery_hours'],
                    cognitive_load=data.get('cognitive_load', 0.5),  # Default if missing
                    embedding=self._parse_embedding(data.get('embedding')),
                    created_at=data['created_at'],
                    updated_at=data['updated_at']
                ))
            
            return task_types
            
        except Exception as e:
            print(f"Error getting user task types: {e}")
            return []
    
    async def update_task_type_habits(self, task_type_id: str,
                                     weekly_habit_scores: List[float],
                                     completion_count: int):
        """Update the weekly habit scores and completion count for a task type"""
        try:
            result = self.supabase.table("task_types")\
                .update({
                    "weekly_habit_scores": weekly_habit_scores,
                    "completion_count": completion_count,
                    "updated_at": datetime.now().isoformat()
                })\
                .eq("id", task_type_id)\
                .execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Error updating task type habits: {e}")
            return None
    
    async def increment_completions_and_check_for_update(self, task_type_id: str, 
                                                       day_of_week: int, 
                                                       hour: int, 
                                                       success: bool) -> bool:
        """Increment completion count and slot confidence, check if mem0 update needed (after 5 completions)"""
        try:
            print(f"📈 Incrementing completion for task type {task_type_id[:8]}...")
            print(f"   📅 Day: {day_of_week}, Hour: {hour}, Success: {success}")
            
            # Get current task type data
            result = self.supabase.table("task_types")\
                .select("completion_count, completions_since_last_update, slot_confidence")\
                .eq("id", task_type_id)\
                .execute()
            
            if not result.data:
                print(f"❌ Task type {task_type_id[:8]} not found")
                return False
            
            data = result.data[0]
            new_completion_count = data['completion_count'] + 1
            new_completions_since_update = data['completions_since_last_update'] + 1
            
            print(f"   📊 Previous completions: {data['completion_count']} → {new_completion_count}")
            print(f"   🔄 Since last update: {data['completions_since_last_update']} → {new_completions_since_update}")
            
            # Update slot confidence for successful completions
            slot_confidence = data['slot_confidence']
            if slot_confidence:
                current_confidence = slot_confidence[day_of_week][hour]
                if success:
                    # Increase confidence for successful completion
                    new_confidence = current_confidence + 0.1
                else:
                    # Decrease confidence for failed completion
                    new_confidence = current_confidence - 0.05
                
                # Regularization: prevent extreme values
                if new_confidence > 0.8:
                    new_confidence = new_confidence * 0.95 + 0.5 * 0.05
                elif new_confidence < 0.2:
                    new_confidence = new_confidence * 0.95 + 0.5 * 0.05
                
                slot_confidence[day_of_week][hour] = max(0.0, min(1.0, new_confidence))
            
            # Update the database
            print(f"   💾 Updating database with new completion data...")
            self.supabase.table("task_types")\
                .update({
                    "completion_count": new_completion_count,
                    "completions_since_last_update": new_completions_since_update,
                    "slot_confidence": slot_confidence
                })\
                .eq("id", task_type_id)\
                .execute()
            
            # Return True if we've hit the threshold for mem0 update (5 completions)
            needs_mem0_update = new_completions_since_update >= 5
            if needs_mem0_update:
                print(f"🎯 Task type ready for Mem0 update! ({new_completions_since_update} completions)")
            else:
                print(f"   ✅ Completion updated successfully ({5 - new_completions_since_update} more needed for Mem0 sync)")
            
            return needs_mem0_update
            
        except Exception as e:
            print(f"❌ Warning: Could not increment completion count: {e}")
            return False
    
    async def get_task_types_needing_mem0_update(self, user_id: str) -> List[str]:
        """Get task types that have 5+ completions since last update (need mem0 sync)"""
        try:
            result = self.supabase.table("task_types")\
                .select("id")\
                .eq("user_id", user_id)\
                .gte("completions_since_last_update", 5)\
                .execute()
            
            return [row['id'] for row in result.data] if result.data else []
        except Exception as e:
            print(f"Error getting task types needing mem0 update: {e}")
            return []
    
    async def reset_completions_since_last_update(self, task_type_id: str):
        """Reset completions counter after mem0 update is completed"""
        try:
            self.supabase.table("task_types")\
                .update({
                    "completions_since_last_update": 0,
                    "last_mem0_update": datetime.now().isoformat()
                })\
                .eq("id", task_type_id)\
                .execute()
        except Exception as e:
            print(f"Error resetting completion count: {e}")
    
    async def update_task_type_from_mem0(self, task_type_id: str, 
                                        importance_score: float = None,
                                        recovery_hours: float = None):
        """Update task type with insights from mem0"""
        try:
            updates = {}
            if importance_score is not None:
                updates["importance_score"] = importance_score
            if recovery_hours is not None:
                updates["recovery_hours"] = recovery_hours
            
            if updates:
                self.supabase.table("task_types")\
                    .update(updates)\
                    .eq("id", task_type_id)\
                    .execute()
                    
        except Exception as e:
            print(f"Error updating task type from mem0: {e}") 