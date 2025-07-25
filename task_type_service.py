"""
Task Type Service - Tier 2 Implementation
Handles task type creation, vector search, and similarity matching
"""

import openai
import uuid
from typing import List, Optional, Dict, Any
from supabase import Client
from models import TaskType, TaskCategory, TaskTypeSimilarity, initialize_neutral_arrays
from datetime import datetime


class TaskTypeService:
    def __init__(self, supabase: Client, openai_api_key: str):
        self.supabase = supabase
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for task description"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    async def find_similar_task_type(self, user_id: str, 
                                   description: str, 
                                   threshold: float = 0.8) -> Optional[TaskTypeSimilarity]:
        """Find similar task type using vector search"""
        embedding = self.generate_embedding(description)
        
        try:
            # Use Supabase vector similarity search
            result = self.supabase.rpc(
                'match_task_types',
                {
                    'query_embedding': embedding,
                    'match_threshold': threshold,
                    'match_count': 1,
                    'target_user_id': user_id
                }
            ).execute()
            
            if result.data:
                match = result.data[0]
                
                # Convert to TaskType model
                task_type = TaskType(
                    id=match['id'],
                    user_id=uuid.UUID(user_id),
                    task_type=match['task_type'],
                    category=TaskCategory(match['category']),
                    hourly_scores=match['hourly_scores'],
                    confidence_scores=match['confidence_scores'],
                    performance_by_hour=match['performance_by_hour'],
                    cognitive_load=match['cognitive_load'],
                    recovery_hours=match['recovery_hours'],
                    typical_duration=match['typical_duration'],
                    importance_score=match['importance_score'],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                return TaskTypeSimilarity(
                    task_type=task_type,
                    similarity=match['similarity']
                )
                
        except Exception as e:
            print(f"Error in vector search: {e}")
            
        return None
    
    async def create_task_type(self, user_id: str, 
                             task_type: str, 
                             category: TaskCategory,
                             description: Optional[str] = None) -> TaskType:
        """Create new task type with neutral patterns"""
        
        # Generate embedding for the task type
        embedding_text = f"{task_type} {description or ''}"
        embedding = self.generate_embedding(embedding_text)
        
        # Initialize neutral patterns
        hourly_scores, confidence_scores, performance_by_hour = initialize_neutral_arrays()
        
        # Set category-specific defaults
        cognitive_load = self._get_category_cognitive_load(category)
        recovery_hours = self._get_category_recovery_hours(category)
        typical_duration = self._get_category_typical_duration(category)
        
        new_task_type = {
            "user_id": user_id,
            "task_type": task_type,
            "category": category.value,
            "hourly_scores": hourly_scores,
            "confidence_scores": confidence_scores,
            "performance_by_hour": performance_by_hour,
            "cognitive_load": cognitive_load,
            "recovery_hours": recovery_hours,
            "typical_duration": typical_duration,
            "importance_score": 0.5,  # Neutral importance initially
            "embedding": embedding
        }
        
        try:
            result = self.supabase.table("task_types").insert(new_task_type).execute()
            data = result.data[0]
            
            return TaskType(
                id=data['id'],
                user_id=uuid.UUID(user_id),
                task_type=data['task_type'],
                category=TaskCategory(data['category']),
                hourly_scores=data['hourly_scores'],
                confidence_scores=data['confidence_scores'],
                performance_by_hour=data['performance_by_hour'],
                cognitive_load=data['cognitive_load'],
                recovery_hours=data['recovery_hours'],
                typical_duration=data['typical_duration'],
                importance_score=data['importance_score'],
                embedding=data['embedding'],
                created_at=data['created_at'],
                updated_at=data['updated_at']
            )
            
        except Exception as e:
            print(f"Error creating task type: {e}")
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
                    category=TaskCategory(data['category']),
                    hourly_scores=data['hourly_scores'],
                    confidence_scores=data['confidence_scores'],
                    performance_by_hour=data['performance_by_hour'],
                    cognitive_load=data['cognitive_load'],
                    recovery_hours=data['recovery_hours'],
                    typical_duration=data['typical_duration'],
                    importance_score=data['importance_score'],
                    embedding=data['embedding'],
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
                    category=TaskCategory(data['category']),
                    hourly_scores=data['hourly_scores'],
                    confidence_scores=data['confidence_scores'],
                    performance_by_hour=data['performance_by_hour'],
                    cognitive_load=data['cognitive_load'],
                    recovery_hours=data['recovery_hours'],
                    typical_duration=data['typical_duration'],
                    importance_score=data['importance_score'],
                    embedding=data['embedding'],
                    created_at=data['created_at'],
                    updated_at=data['updated_at']
                ))
            
            return task_types
            
        except Exception as e:
            print(f"Error getting user task types: {e}")
            return []
    
    async def update_task_type_arrays(self, task_type_id: str,
                                    hourly_scores: List[float],
                                    confidence_scores: List[float],
                                    performance_by_hour: List[float]):
        """Update the learned pattern arrays for a task type"""
        try:
            result = self.supabase.table("task_types")\
                .update({
                    "hourly_scores": hourly_scores,
                    "confidence_scores": confidence_scores,
                    "performance_by_hour": performance_by_hour,
                    "updated_at": datetime.now().isoformat()
                })\
                .eq("id", task_type_id)\
                .execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Error updating task type arrays: {e}")
            return None
    
    def _get_category_cognitive_load(self, category: TaskCategory) -> float:
        """Get default cognitive load based on category"""
        defaults = {
            TaskCategory.FOCUSED: 0.8,      # High cognitive load
            TaskCategory.COLLABORATIVE: 0.6, # Medium cognitive load
            TaskCategory.ADMINISTRATIVE: 0.4  # Lower cognitive load
        }
        return defaults.get(category, 0.5)
    
    def _get_category_recovery_hours(self, category: TaskCategory) -> float:
        """Get default recovery hours based on category"""
        defaults = {
            TaskCategory.FOCUSED: 1.5,      # Need more recovery
            TaskCategory.COLLABORATIVE: 0.5, # Medium recovery
            TaskCategory.ADMINISTRATIVE: 0.25 # Quick recovery
        }
        return defaults.get(category, 0.5)
    
    def _get_category_typical_duration(self, category: TaskCategory) -> float:
        """Get default duration based on category"""
        defaults = {
            TaskCategory.FOCUSED: 2.0,      # Longer focused sessions
            TaskCategory.COLLABORATIVE: 1.0, # Standard meeting length
            TaskCategory.ADMINISTRATIVE: 0.5  # Quick admin tasks
        }
        return defaults.get(category, 1.0)
    
    def categorize_task_description(self, description: str) -> TaskCategory:
        """Automatically categorize task based on description"""
        description_lower = description.lower()
        
        # Focused work keywords
        focused_keywords = [
            'code', 'coding', 'programming', 'development', 'study', 'studying',
            'research', 'analysis', 'design', 'writing', 'deep', 'focus',
            'concentration', 'learn', 'learning', 'project', 'assignment'
        ]
        
        # Collaborative keywords
        collaborative_keywords = [
            'meeting', 'call', 'discussion', 'sync', 'standup', 'review',
            'team', 'collaboration', 'interview', 'presentation', 'demo',
            'brainstorm', 'planning', 'ceremony', 'retrospective'
        ]
        
        # Administrative keywords
        administrative_keywords = [
            'email', 'admin', 'administrative', 'filing', 'organize', 
            'schedule', 'calendar', 'booking', 'expense', 'report',
            'documentation', 'paperwork', 'form', 'update'
        ]
        
        # Count keyword matches
        focused_score = sum(1 for keyword in focused_keywords if keyword in description_lower)
        collaborative_score = sum(1 for keyword in collaborative_keywords if keyword in description_lower)
        administrative_score = sum(1 for keyword in administrative_keywords if keyword in description_lower)
        
        # Return category with highest score
        if focused_score >= collaborative_score and focused_score >= administrative_score:
            return TaskCategory.FOCUSED
        elif collaborative_score >= administrative_score:
            return TaskCategory.COLLABORATIVE
        else:
            return TaskCategory.ADMINISTRATIVE 