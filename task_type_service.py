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

import os
from dotenv import load_dotenv
load_dotenv()


class TaskTypeService:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
    
    def generate_embedding(self, task_type: str) -> List[float]:
        """Generate OpenAI embedding from task type name only"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=task_type.strip()
            )
            embedding = response.data[0].embedding
            print(f"   🧮 Generated embedding: first 3 values = {embedding[:3]}")
            return embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise  # Don't return zeros, let it fail
    
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
        
        print(f"🔍 RAG Search: '{task_type}' (threshold: {threshold:.2f})")
        
        embedding = self.generate_embedding(task_type)
        
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
                             description: Optional[str] = None) -> TaskType:
        """Create new task type with LLM-analyzed behavioral patterns"""
        
        print(f"🆕 Creating new task type: '{task_type}' for user {user_id[:8]}...")
        if description:
            print(f"   📝 Description: {description}")
        
        # Generate embedding from task type name only
        embedding = self.generate_embedding(task_type)
        print(f"   🧮 Generated embedding (dimensions: {len(embedding)})")
        
        # Use LLM to analyze task and generate behavioral patterns
        print(f"   🤖 Analyzing task with LLM to generate behavioral patterns...")
        pattern_analysis = await self._analyze_task_patterns_with_llm(task_type, description)
        
        weekly_habit_scores = pattern_analysis['weekly_habit_scores']
        slot_confidence = pattern_analysis['slot_confidence']
        cognitive_load = pattern_analysis['cognitive_load']
        typical_duration = pattern_analysis['typical_duration']
        importance_score = pattern_analysis['importance_score']
        recovery_hours = pattern_analysis['recovery_hours']
        
        print(f"   📊 LLM generated patterns:")
        print(f"      🧠 Cognitive load: {cognitive_load:.2f}")
        print(f"      ⏱️ Typical duration: {typical_duration:.1f}h")
        print(f"      ⭐ Importance: {importance_score:.2f}")
        print(f"      🔄 Recovery: {recovery_hours:.1f}h")
        
        new_task_type = {
            "user_id": user_id,
            "task_type": task_type,
            "description": description,
            "weekly_habit_scores": weekly_habit_scores,
            "slot_confidence": slot_confidence,
            "completion_count": 0,
            "completions_since_last_update": 0,
            "typical_duration": typical_duration,
            "importance_score": importance_score,
            "recovery_hours": recovery_hours,
            "cognitive_load": cognitive_load,
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

    async def _analyze_task_patterns_with_llm(self, task_type: str, description: Optional[str] = None) -> Dict:
        """Use LLM to analyze task and generate behavioral patterns"""
        
        function_schema = {
            "name": "analyze_task_behavioral_patterns",
            "description": "Analyze a task type and generate behavioral scheduling patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_patterns": {
                        "type": "string",
                        "description": "Compact time patterns: 'days:hour_start-hour_end:preference_score'. Days: 0-6=Mon-Sun, 0-4=weekdays, 5-6=weekend. Example: '0-4:9-17:0.8,5-6:10-14:0.6'"
                    },
                    "base_confidence": {
                        "type": "number",
                        "description": "Base confidence level for all time slots (0.0-1.0). Will be used to initialize the entire 7×24 confidence matrix",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "cognitive_load": {
                        "type": "number",
                        "description": "Cognitive demand of the task (0.0=very easy, 1.0=very demanding)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "typical_duration": {
                        "type": "number",
                        "description": "Expected duration in hours for this type of task",
                        "minimum": 0.25,
                        "maximum": 8.0
                    },
                    "importance_score": {
                        "type": "number",
                        "description": "General importance level (0.0=low priority, 1.0=critical)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "recovery_hours": {
                        "type": "number",
                        "description": "Buffer time needed after completing this task type (in hours)",
                        "minimum": 0.0,
                        "maximum": 4.0
                    },
                    "analysis_reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the behavioral pattern analysis"
                    }
                },
                "required": ["time_patterns", "base_confidence", "cognitive_load", "typical_duration", "importance_score", "recovery_hours", "analysis_reasoning"]
            }
        }
        
        context = f"""
Analyze the following task type and generate behavioral scheduling patterns:

TASK TYPE: "{task_type}"
DESCRIPTION: "{description or 'No description provided'}"

🔍 IMPORTANT: Carefully analyze the DESCRIPTION for time preferences like:
- "morning", "afternoon", "evening" 
- "prefer in morning", "focus", "creative hours"
- Any specific time mentions or scheduling hints
- Energy level indicators ("when I'm most focused", "peak energy")

Please analyze this task and generate:

1. TIME_PATTERNS: Compact string format for scheduling preferences
   Format: "days:hour_start-hour_end:preference_score,days:hour_start-hour_end:preference_score"
   - Days: 0-6 (Mon-Sun), 0-4 (weekdays), 5-6 (weekend), or specific like 0,2,4
   - Hours: 0-23 (24-hour format)
   - Preference score: 0.0-1.0 (higher = better time for this task)
   
   Examples:
   - Business meeting: "0-4:9-17:0.8" (weekdays 9AM-5PM)
   - Workout: "0-6:6-8:0.9,0-6:18-20:0.8" (morning and evening)
   - Study session: "0-6:8-11:0.9,0-6:20-22:0.7" (morning focus + evening)
   - Morning preference: "0-6:6-11:0.9,0-6:14-17:0.5" (strong morning preference)
   - Creative tasks: "0-6:8-12:0.9,0-6:20-22:0.7" (morning creativity + evening)

2. BASE_CONFIDENCE: Single confidence level (0.0-1.0) that will be applied to all time slots
   - Start moderate (0.3-0.5) for new task types
   - Higher confidence for well-defined task types

3. TASK CHARACTERISTICS:
   - Cognitive load: How mentally demanding is this task?
   - Typical duration: How long does this type of task usually take?
   - Importance score: General priority level for this task type
   - Recovery hours: Buffer time needed after completion

GUIDELINES:
- Morning (6-11): Focus work, exercise, planning, high cognitive tasks
- Afternoon (12-17): Meetings, routine work, administrative tasks
- Evening (18-22): Creative work, learning, personal tasks
- Night (23-5): Generally avoid except for specific cases

⚠️ CRITICAL: If description mentions time preferences, MUST adjust time patterns accordingly!
Example: "prefer in morning" → high scores for 6-11, lower for afternoon
Example: "creative tasks in morning hours" → 0-6:8-12:0.9

Be specific! Generate patterns that match the described preferences.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a behavioral scheduling analyst. Analyze task types and generate realistic behavioral patterns for optimal scheduling. Consider cognitive demands, typical timing preferences, and energy requirements."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                functions=[function_schema],
                function_call={"name": "analyze_task_behavioral_patterns"},
                temperature=0.3
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "analyze_task_behavioral_patterns":
                import json
                result = json.loads(function_call.arguments)
                
                print(f"   🤖 LLM Analysis: {result.get('analysis_reasoning', 'No reasoning provided')}")
                
                # Parse compact format and expand into full arrays
                time_patterns = result.get('time_patterns', '')
                base_confidence = result.get('base_confidence', 0.4)
                
                print(f"   📋 Time patterns: '{time_patterns}'")
                print(f"   🎯 Base confidence: {base_confidence:.2f}")
                
                # Expand time patterns into 168-element weekly habit scores
                weekly_scores = self._expand_time_patterns_to_weekly_scores(time_patterns)
                if not weekly_scores:
                    print(f"   ⚠️ Could not parse time patterns, using fallback")
                    weekly_scores = self._generate_fallback_weekly_scores(task_type)
                else:
                    print(f"   ✅ Successfully parsed time patterns into weekly scores")
                    # Show some sample scores for verification
                    sample_morning = [weekly_scores[i*24 + 8] for i in range(7)]  # 8 AM each day
                    sample_afternoon = [weekly_scores[i*24 + 14] for i in range(7)]  # 2 PM each day
                    print(f"   📊 Morning (8AM) scores: {[f'{s:.1f}' for s in sample_morning]}")
                    print(f"   📊 Afternoon (2PM) scores: {[f'{s:.1f}' for s in sample_afternoon]}")
                
                # Create confidence matrix from base confidence
                slot_conf = self._create_confidence_matrix_from_base(base_confidence)
                
                return {
                    'weekly_habit_scores': weekly_scores,
                    'slot_confidence': slot_conf,
                    'cognitive_load': max(0.0, min(1.0, result.get('cognitive_load', 0.5))),
                    'typical_duration': max(0.25, min(8.0, result.get('typical_duration', 1.0))),
                    'importance_score': max(0.0, min(1.0, result.get('importance_score', 0.5))),
                    'recovery_hours': max(0.0, min(4.0, result.get('recovery_hours', 0.5))),
                    'reasoning': result.get('analysis_reasoning', '')
                }
            else:
                raise ValueError("LLM did not return expected function call")
                
        except Exception as e:
            print(f"❌ Error analyzing task patterns with LLM: {e}")
            print(f"   🔄 Using fallback pattern generation...")
            return self._generate_fallback_patterns(task_type)
    
    
    def _generate_fallback_slot_confidence(self) -> List[List[float]]:
        """Generate fallback slot confidence matrix"""
        # 7 days x 24 hours, moderate confidence
        return [[0.4 for _ in range(24)] for _ in range(7)]
    
    def _generate_fallback_weekly_scores(self, task_type: str) -> List[float]:
        """Generate fallback weekly habit scores when parsing fails"""
        return [0.4] * 168  # 7 days × 24 hours, neutral baseline
    
    def _generate_fallback_patterns(self, task_type: str) -> Dict:
        """Generate complete fallback patterns when LLM fails"""
        return {
            'weekly_habit_scores': [0.4] * 168,
            'slot_confidence': self._generate_fallback_slot_confidence(),
            'cognitive_load': 0.5,
            'typical_duration': 1.0,
            'importance_score': 0.5,
            'recovery_hours': 0.5,
            'reasoning': f'Fallback pattern generation for task type: {task_type}'
        }
    
    def _expand_time_patterns_to_weekly_scores(self, time_patterns: str) -> Optional[List[float]]:
        """Expand compact time patterns string into 168-element weekly habit scores array"""
        if not time_patterns or not time_patterns.strip():
            return None
        
        # Start with neutral baseline
        weekly_scores = [0.4] * 168  # 7 days × 24 hours
        
        # Parse time patterns using existing method
        parsed_patterns = self._parse_time_pattern_string(time_patterns)
        if not parsed_patterns:
            return None
        
        # Apply each pattern to the weekly scores array
        for pattern in parsed_patterns:
            days = pattern.get("days", [])
            hour_start = pattern.get("hour_start")
            hour_end = pattern.get("hour_end")
            preference_score = pattern.get("boost", 0.5)  # 'boost' is the preference score
            
            # Validate pattern
            if (not days or hour_start is None or hour_end is None or 
                not all(0 <= d <= 6 for d in days) or 
                not (0 <= hour_start <= 23) or not (0 <= hour_end <= 23)):
                continue
            
            # Apply preference score to all day/hour combinations in the range
            for day in days:
                for hour in range(hour_start, hour_end + 1):  # inclusive range
                    weekly_index = day * 24 + hour
                    if weekly_index < len(weekly_scores):
                        weekly_scores[weekly_index] = preference_score
        
        return weekly_scores
    
    def _create_confidence_matrix_from_base(self, base_confidence: float) -> List[List[float]]:
        """Create 7×24 confidence matrix from single base confidence value"""
        # Clamp base confidence to valid range
        confidence = max(0.0, min(1.0, base_confidence))
        
        # Create 7 days × 24 hours matrix with same confidence value
        return [[confidence for _ in range(24)] for _ in range(7)]
    
    def _parse_time_pattern_string(self, pattern_string: str) -> List[Dict]:
        """Parse compact time pattern string into structured format
        
        Format: "days:hour_start-hour_end:boost,days:hour_start-hour_end:boost"
        Example: "0-6:6-11:0.8,5-6:17-21:0.9"
        """
        patterns = []
        
        if not pattern_string or not pattern_string.strip():
            return patterns

        # Split by comma to get individual patterns
        for pattern_part in pattern_string.split(','):
            pattern_part = pattern_part.strip()
            if not pattern_part:
                continue
                
            try:
                # Split by colon: days:hour_start-hour_end:boost
                parts = pattern_part.split(':')
                if len(parts) != 3:
                    continue
                    
                days_part, hours_part, boost_part = parts
                
                # Parse days (can be range like "0-6" or specific like "0,2,4")
                days = []
                if '-' in days_part:
                    # Range format like "0-6"
                    start_day, end_day = map(int, days_part.split('-'))
                    days = list(range(start_day, end_day + 1))
                else:
                    # Specific days like "0,2,4" or single day "0"
                    days = [int(d.strip()) for d in days_part.split(',')]
                
                # Parse hours (format: "start-end")
                hour_start, hour_end = map(int, hours_part.split('-'))
                
                # Parse boost
                boost = float(boost_part)
                
                # Validate ranges
                if (all(0 <= d <= 6 for d in days) and 
                    0 <= hour_start <= 23 and 0 <= hour_end <= 23 and
                    0.0 <= boost <= 1.0):
                    
                    patterns.append({
                        "days": days,
                        "hour_start": hour_start,
                        "hour_end": hour_end,
                        "boost": boost
                    })
                    
            except (ValueError, IndexError) as e:
                print(f"⚠️ Could not parse pattern part '{pattern_part}': {e}")
                continue
        
        return patterns
    
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