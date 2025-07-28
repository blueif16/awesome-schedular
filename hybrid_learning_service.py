"""
Hybrid Learning Service - Advanced Pattern Learning
Combines real-time behavioral updates with periodic LLM interpretation from Mem0
"""

import asyncio
import json
import openai
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from task_type_service import TaskTypeService
from models import get_weekly_index, get_day_hour_from_index


@dataclass
class BehavioralUpdate:
    """Queue item for real-time behavioral updates"""
    task_type_id: str
    day_of_week: int  # 0=Sunday, 6=Saturday
    hour: int         # 0-23
    success_rating: float  # 0.0-1.0
    energy_after: Optional[float] = None
    user_id: str = None
    timestamp: datetime = None


@dataclass
class PatternAdjustment:
    """LLM-suggested pattern adjustment"""
    day_pattern: str  # "all", "weekdays", "weekends", or specific day like "monday"
    hour_range: List[int]  # [9, 10, 11] for 9-11am
    modifier: float    # 1.2 = boost by 20%, 0.8 = reduce by 20%
    reason: str       # Human-readable explanation
    task_pattern: str = ".*"  # Regex to match task types


class HybridLearningService:
    def __init__(self, task_type_service: TaskTypeService, 
                 memory_service, openai_api_key: str):
        self.task_type_service = task_type_service
        self.memory_service = memory_service
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Learning parameters
        self.behavioral_weight = 0.7    # Trust behavior more
        self.llm_weight = 0.3          # LLM provides context
        self.base_learning_rate = 0.2  # How fast to learn from behavior
        
        # Simple queue for behavioral updates
        self.behavioral_queue = []
    
    def queue_behavioral_update(self, update: BehavioralUpdate):
        """Queue a behavioral update for real-time processing"""
        if update.timestamp is None:
            update.timestamp = datetime.now()
        self.behavioral_queue.append(update)
        
        # Process immediately for responsiveness
        asyncio.create_task(self._process_behavioral_update(update))
    
    async def _process_behavioral_update(self, update: BehavioralUpdate):
        """Process single behavioral update immediately"""
        try:
            # Get current task type
            task_type = await self.task_type_service.get_task_type(update.task_type_id)
            if not task_type:
                print(f"Task type {update.task_type_id} not found")
                return
            
            # Calculate weekly array index
            weekly_index = get_weekly_index(update.day_of_week, update.hour)
            
            # Update habit score with adaptive learning rate
            current_score = task_type.weekly_habit_scores[weekly_index]
            learning_rate = self._calculate_learning_rate(task_type.completion_count)
            
            # Simple success signal (can be enhanced)
            signal = update.success_rating
            new_score = current_score * (1 - learning_rate) + signal * learning_rate
            new_score = max(0.0, min(1.0, new_score))  # Clamp to [0,1]
            
            # Update the array
            updated_habit_scores = task_type.weekly_habit_scores.copy()
            updated_habit_scores[weekly_index] = new_score
            
            # Save to database
            await self.task_type_service.update_task_type_habits(
                update.task_type_id,
                updated_habit_scores,
                task_type.completion_count + 1
            )
            
            # Mark behavioral update timestamp
            await self._mark_behavioral_update(update.task_type_id)
            
            # Add insight to Mem0 if significant pattern
            await self._maybe_add_behavioral_insight(update, task_type, new_score)
            
            print(f"✅ Behavioral update: {task_type.task_type} at {self._format_time(update.day_of_week, update.hour)}")
            print(f"   Score: {current_score:.2f} → {new_score:.2f}")
            
        except Exception as e:
            print(f"❌ Error processing behavioral update: {e}")
    
    def _calculate_learning_rate(self, completion_count: int) -> float:
        """Adaptive learning rate - learn faster with fewer completions"""
        return self.base_learning_rate / (1 + completion_count * 0.05)
    
    async def _mark_behavioral_update(self, task_type_id: str):
        """Mark when this task type was last updated behaviorally"""
        try:
            await self.task_type_service.supabase.table("task_types")\
                .update({"last_behavioral_update": datetime.now().isoformat()})\
                .eq("id", task_type_id)\
                .execute()
        except Exception as e:
            print(f"Warning: Could not mark behavioral update: {e}")
    
    async def _maybe_add_behavioral_insight(self, update: BehavioralUpdate, 
                                          task_type, new_score: float):
        """Add insight to Mem0 if a significant pattern is detected"""
        
        # Only add insights for strong patterns
        if new_score > 0.8 or new_score < 0.2:
            day_name = self._get_day_name(update.day_of_week)
            time_str = f"{update.hour:02d}:00"
            
            if new_score > 0.8:
                insight = (f"User consistently performs well with '{task_type.task_type}' "
                          f"on {day_name} at {time_str} (success rate: {update.success_rating:.1%})")
            else:
                insight = (f"User struggles with '{task_type.task_type}' "
                          f"on {day_name} at {time_str} (success rate: {update.success_rating:.1%})")
            
            await self.memory_service.add(
                messages=[{
                    "role": "assistant",
                    "content": insight
                }],
                user_id=update.user_id,
                metadata={
                    "category": "behavioral_insight",
                    "type": "behavioral_pattern",
                    "task_type_id": update.task_type_id,
                    "day_of_week": update.day_of_week,
                    "hour": update.hour,
                    "score": new_score,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def sync_task_types_needing_update(self, user_id: str):
        """Completion-based sync: update task types that have 5+ completions since last update"""
        
        try:
            # 1. Get task types that need mem0 update (5+ completions)
            task_type_ids_needing_update = await self.task_type_service.get_task_types_needing_mem0_update(user_id)
            
            if not task_type_ids_needing_update:
                print(f"   No task types need mem0 update for user {user_id}")
                return
            
            print(f"🧠 Mem0 sync for {len(task_type_ids_needing_update)} task types with 5+ completions")
            
            # 2. Get insights from AsyncMemory
            insights = await self.memory_service.search(
                query="importance preferences buffer time recovery energy habits productive patterns timing",
                user_id=user_id,
                limit=30
            )
            
            # 3. Process each task type needing update
            for task_type_id in task_type_ids_needing_update:
                try:
                    task_type = await self.task_type_service.get_task_type(task_type_id)
                    if task_type:
                        # Get mem0 insights and update importance/recovery
                        await self._update_task_type_from_mem0_insights(
                            task_type.__dict__, insights, user_id
                        )
                        
                        # Reset the completion counter after update
                        await self.task_type_service.reset_completions_since_last_update(task_type_id)
                        
                except Exception as e:
                    print(f"   Error syncing task type {task_type_id}: {e}")
            
            print(f"✅ Mem0 sync completed for user {user_id}")
            
        except Exception as e:
            print(f"❌ Error in mem0 sync: {e}")
    
    async def check_and_trigger_mem0_update_if_needed(self, user_id: str):
        """Check if any task types need mem0 update and trigger if needed"""
        try:
            task_types_needing_update = await self.task_type_service.get_task_types_needing_mem0_update(user_id)
            if task_types_needing_update:
                await self.sync_task_types_needing_update(user_id)
        except Exception as e:
            print(f"Error checking for mem0 update: {e}")
    
    async def _update_task_type_from_mem0_insights(self, task_type_data: Dict, 
                                                  insights: List[Dict], 
                                                  user_id: str):
        """Update task type importance and recovery hours based on mem0 insights"""
        
        try:
            # Build focused prompt for importance and recovery
            prompt = self._build_importance_and_recovery_prompt(task_type_data, insights)
            
            # Get LLM response focused on importance and recovery
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Parse response for importance and recovery values
            result = self._parse_importance_and_recovery_response(response.choices[0].message.content)
            
            if result:
                # Update task type with new values
                await self.task_type_service.update_task_type_from_mem0(
                    task_type_data['id'],
                    importance_score=result.get('importance_score'),
                    recovery_hours=result.get('recovery_hours')
                )
                
                print(f"   ✅ Updated {task_type_data['task_type']}: importance={result.get('importance_score', 'unchanged')}, recovery={result.get('recovery_hours', 'unchanged')}")
                
        except Exception as e:
            print(f"   ❌ Error updating task type from mem0: {e}")
    
    def _build_importance_and_recovery_prompt(self, task_type_data: Dict, insights: List[Dict]) -> str:
        """Build focused prompt for importance and recovery time assessment"""
        
        # Format insights
        insight_text = "\n".join([f"- {insight.get('text', '')}" for insight in insights[:10]])
        
        return f"""
Task: {task_type_data['task_type']}
{f"Description: {task_type_data.get('description')}" if task_type_data.get('description') else ""}
Current importance: {task_type_data.get('importance_score', 0.5)}
Current recovery time: {task_type_data.get('recovery_hours', 0.5)} hours

User insights from memory:
{insight_text or "No relevant insights"}

Based on these insights, assess:
1. How important is this task to the user? (0.0-1.0 scale)
2. How much buffer/recovery time should be scheduled after this task? (0.0-2.0 hours)

Return JSON format:
{{
    "importance_score": 0.8,
    "recovery_hours": 1.2,
    "reasoning": "User mentioned this is critical for deadlines, needs focus time after"
}}

Only suggest changes if there's clear evidence in the insights.
"""
    
    def _parse_importance_and_recovery_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM response for importance and recovery values"""
        try:
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Validate and clamp values
                parsed_result = {}
                if 'importance_score' in result:
                    parsed_result['importance_score'] = max(0.0, min(1.0, float(result['importance_score'])))
                if 'recovery_hours' in result:
                    parsed_result['recovery_hours'] = max(0.0, min(2.0, float(result['recovery_hours'])))
                
                return parsed_result if parsed_result else None
                
        except Exception as e:
            print(f"   Warning: Could not parse LLM response: {e}")
            
        return None
    
    async def _sync_task_type_from_insights(self, task_type_data: Dict, 
                                          insights: List[Dict], user_id: str):
        """Use LLM to interpret insights and update task type patterns"""
        
        # Get recent completions for context
        recent_completions = await self._get_recent_completions(task_type_data['id'])
        
        # Prepare LLM prompt
        prompt = self._build_interpretation_prompt(
            task_type_data, insights, recent_completions
        )
        
        try:
            # Get LLM interpretation
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # Parse adjustments
            adjustments = self._parse_llm_adjustments(response.choices[0].message.content)
            
            if adjustments:
                # Apply adjustments to habit array
                await self._apply_pattern_adjustments(
                    task_type_data['id'], 
                    task_type_data['weekly_habit_scores'], 
                    adjustments
                )
                
                print(f"   🎯 Applied {len(adjustments)} LLM adjustments to {task_type_data['task_type']}")
            
        except Exception as e:
            print(f"   ❌ LLM interpretation failed: {e}")
    
    def _build_interpretation_prompt(self, task_type_data: Dict, 
                                   insights: List[Dict], 
                                   recent_completions: List[Dict]) -> str:
        """Build prompt for LLM to interpret insights"""
        
        # Summarize current pattern
        current_pattern = self._summarize_weekly_pattern(task_type_data['weekly_habit_scores'])
        
        # Format insights
        insight_text = "\n".join([
            f"- {insight.get('text', insight.get('content', 'Unknown insight'))}"
            for insight in insights[:10]  # Limit to most relevant
        ])
        
        # Format recent completions
        completion_text = "\n".join([
            f"- {comp['day']} {comp['time']}: {comp['success']:.1%} success, {comp.get('energy', 'N/A')} energy"
            for comp in recent_completions[:10]
        ])
        
        return f"""
Task: {task_type_data['task_type']}
{f"Description: {task_type_data.get('description')}" if task_type_data.get('description') else ""}
Current pattern: {current_pattern}

Recent completions:
{completion_text or "No recent completions"}

User insights:
{insight_text or "No relevant insights"}

Based on these insights, suggest adjustments to the weekly habit pattern.
Return a JSON array like this:
[
    {{
        "day_pattern": "monday",
        "hour_range": [9, 10, 11],
        "modifier": 1.2,
        "reason": "User mentioned loving Monday mornings"
    }}
]

Valid day_pattern: "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "weekdays", "weekends", "all"
Modifier: >1.0 = boost, <1.0 = reduce, 1.0 = no change
Only suggest if there's clear evidence.
"""
    
    def _summarize_weekly_pattern(self, weekly_scores: List[float]) -> str:
        """Create human-readable summary of weekly pattern"""
        if not weekly_scores or len(weekly_scores) != 168:
            return "No pattern data"
        
        # Find peak times
        peak_threshold = 0.7
        peaks = []
        
        for i, score in enumerate(weekly_scores):
            if score > peak_threshold:
                day, hour = get_day_hour_from_index(i)
                day_name = self._get_day_name(day)
                peaks.append(f"{day_name} {hour:02d}:00")
        
        if peaks:
            return f"Peak times: {', '.join(peaks[:5])}"
        else:
            return "No strong patterns yet"
    
    def _parse_llm_adjustments(self, llm_response: str) -> List[PatternAdjustment]:
        """Parse LLM response into structured adjustments"""
        try:
            # Extract JSON from response
            start = llm_response.find('[')
            end = llm_response.rfind(']') + 1
            json_str = llm_response[start:end]
            
            adjustments_data = json.loads(json_str)
            
            adjustments = []
            for adj_data in adjustments_data:
                adjustments.append(PatternAdjustment(
                    day_pattern=adj_data['day_pattern'],
                    hour_range=adj_data['hour_range'],
                    modifier=adj_data['modifier'],
                    reason=adj_data['reason'],
                    task_pattern=adj_data.get('task_pattern', '.*')
                ))
            
            return adjustments
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"   Warning: Could not parse LLM adjustments: {e}")
            return []
    
    async def _apply_pattern_adjustments(self, task_type_id: str, 
                                       current_scores: List[float],
                                       adjustments: List[PatternAdjustment]):
        """Apply LLM-suggested adjustments to habit pattern"""
        
        updated_scores = current_scores.copy()
        
        for adjustment in adjustments:
            # Get days to apply to
            target_days = self._get_days_for_pattern(adjustment.day_pattern)
            
            # Apply to each target day and hour range
            for day in target_days:
                for hour in adjustment.hour_range:
                    if 0 <= hour <= 23:  # Valid hour
                        weekly_index = get_weekly_index(day, hour)
                        if 0 <= weekly_index < 168:  # Valid index
                            # Apply modifier with blending
                            current = updated_scores[weekly_index]
                            adjusted = current * adjustment.modifier
                            
                            # Blend with behavioral data (respect behavioral weight)
                            updated_scores[weekly_index] = (
                                current * self.behavioral_weight + 
                                adjusted * self.llm_weight
                            )
                            
                            # Clamp to valid range
                            updated_scores[weekly_index] = max(0.0, min(1.0, updated_scores[weekly_index]))
        
        # Save updated pattern
        try:
            task_type = await self.task_type_service.get_task_type(task_type_id)
            if task_type:
                await self.task_type_service.update_task_type_habits(
                    task_type_id,
                    updated_scores,
                    task_type.completion_count
                )
        except Exception as e:
            print(f"   Error saving adjusted pattern: {e}")
    
    def _get_days_for_pattern(self, day_pattern: str) -> List[int]:
        """Convert day pattern to list of day indices"""
        day_map = {
            "sunday": [0], "monday": [1], "tuesday": [2], "wednesday": [3],
            "thursday": [4], "friday": [5], "saturday": [6],
            "weekdays": [1, 2, 3, 4, 5],  # Mon-Fri
            "weekends": [0, 6],           # Sun, Sat
            "all": list(range(7))         # All days
        }
        return day_map.get(day_pattern.lower(), [])
    
    async def _get_recent_completions(self, task_type_id: str, days: int = 14) -> List[Dict]:
        """Get recent completions for a task type"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            result = await self.task_type_service.supabase.table("events")\
                .select("scheduled_start, success_rating, energy_after")\
                .eq("task_type_id", task_type_id)\
                .eq("completed", True)\
                .gte("scheduled_start", since_date)\
                .order("scheduled_start", desc=True)\
                .limit(20)\
                .execute()
            
            completions = []
            for event in result.data or []:
                start_time = datetime.fromisoformat(event['scheduled_start'])
                completions.append({
                    "day": self._get_day_name(start_time.weekday()),
                    "time": start_time.strftime("%H:%M"),
                    "success": event.get('success_rating', 0.5),
                    "energy": event.get('energy_after')
                })
            
            return completions
            
        except Exception as e:
            print(f"Error getting recent completions: {e}")
            return []
    
    def _get_day_name(self, day_index: int) -> str:
        """Convert day index to name"""
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return days[day_index % 7]
    
    def _format_time(self, day_of_week: int, hour: int) -> str:
        """Format day and hour as readable string"""
        day_name = self._get_day_name(day_of_week)
        return f"{day_name} {hour:02d}:00"
    
    async def handle_task_completion(self, user_id: str, task_type_id: str, 
                                   day_of_week: int, hour: int, success: bool):
        """Called when a task is completed - updates counts and triggers mem0 sync if threshold reached"""
        try:
            # Increment completion count and slot confidence, check if mem0 update needed
            needs_mem0_update = await self.task_type_service.increment_completions_and_check_for_update(
                task_type_id, day_of_week, hour, success
            )
            
            # If we've hit 5 completions, trigger mem0 sync for this user
            if needs_mem0_update:
                await self.check_and_trigger_mem0_update_if_needed(user_id)
            
        except Exception as e:
            print(f"Error in handle_task_completion: {e}") 