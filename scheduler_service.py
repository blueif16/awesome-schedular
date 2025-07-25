"""
Scheduler Service - Core Scheduling Algorithm
Uses Tier 2 patterns to find optimal time slots
"""

import uuid
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from models import TaskType, Event, ScheduleEventRequest, SchedulingResult
from task_type_service import TaskTypeService


class SchedulerService:
    def __init__(self, task_type_service: TaskTypeService):
        self.task_type_service = task_type_service
    
    def calculate_slot_score(self, start_time: datetime, 
                           duration: float, 
                           task_type: TaskType,
                           existing_events: List[Dict] = None) -> float:
        """Calculate comprehensive score for a time slot using Tier 2 patterns"""
        
        hourly_scores = task_type.hourly_scores           # Time preferences
        confidence_scores = task_type.confidence_scores   # Data confidence  
        performance_by_hour = task_type.performance_by_hour  # Energy patterns
        cognitive_load = task_type.cognitive_load
        recovery_hours = task_type.recovery_hours
        
        total_score = 0
        total_weight = 0
        
        # Score each hour the task spans
        current_hour = start_time.hour
        remaining_duration = duration
        hour_scores = []
        
        while remaining_duration > 0:
            hour_fraction = min(1.0, remaining_duration)
            hour_index = current_hour % 24
            
            # Core pattern scores from Tier 2 arrays
            preference = hourly_scores[hour_index]        # When user likes this task
            energy_level = performance_by_hour[hour_index]  # User's energy at this hour
            confidence = confidence_scores[hour_index]    # How sure we are
            
            # Combine preference + energy with confidence weighting
            base_score = (preference * 0.6 + energy_level * 0.4)
            weighted_score = base_score * confidence * hour_fraction
            
            hour_scores.append({
                'hour': current_hour,
                'base_score': base_score,
                'confidence': confidence,
                'weighted_score': weighted_score
            })
            
            total_score += weighted_score
            total_weight += confidence * hour_fraction
            
            remaining_duration -= hour_fraction
            current_hour += 1
        
        # Base score from patterns
        pattern_score = total_score / total_weight if total_weight > 0 else 0.5
        
        # Apply cognitive load penalty if scheduling during low energy
        avg_energy = sum(h['base_score'] for h in hour_scores) / len(hour_scores)
        if cognitive_load > 0.7 and avg_energy < 0.6:
            pattern_score *= 0.7  # Penalty for high-cognitive tasks during low energy
        
        # Check recovery time from previous tasks
        recovery_penalty = self.calculate_recovery_penalty(
            start_time, existing_events or [], recovery_hours
        )
        
        final_score = pattern_score * recovery_penalty
        
        return min(1.0, max(0.1, final_score))  # Clamp between 0.1-1.0
    
    def calculate_recovery_penalty(self, start_time: datetime,
                                 existing_events: List[Dict],
                                 required_recovery_hours: float) -> float:
        """Calculate penalty if not enough recovery time from previous tasks"""
        if not existing_events:
            return 1.0  # No penalty if no existing events
        
        # Find the most recent event before this slot
        recent_events = [
            event for event in existing_events 
            if event['scheduled_end'] <= start_time
        ]
        
        if not recent_events:
            return 1.0  # No recent events
        
        # Get the most recent event
        last_event = max(recent_events, key=lambda e: e['scheduled_end'])
        time_since_last = (start_time - last_event['scheduled_end']).total_seconds() / 3600
        
        # Get cognitive load of previous task (if available)
        prev_cognitive_load = last_event.get('cognitive_load', 0.5)
        
        # Calculate required recovery based on previous task's cognitive load
        required_recovery = required_recovery_hours * prev_cognitive_load
        
        if time_since_last >= required_recovery:
            return 1.0  # Full score - enough recovery time
        else:
            # Partial penalty based on how much recovery time is missing
            recovery_ratio = time_since_last / required_recovery
            return max(0.5, recovery_ratio)  # Minimum 50% score
    
    def is_slot_available(self, start_time: datetime, 
                         duration: float, 
                         existing_events: List[Dict]) -> bool:
        """Check if time slot conflicts with existing events"""
        end_time = start_time + timedelta(hours=duration)
        
        for event in existing_events:
            event_start = event['scheduled_start']
            event_end = event['scheduled_end']
            
            # Check for overlap
            if (start_time < event_end and end_time > event_start):
                return False
        
        return True
    
    def generate_slot_reasoning(self, start_time: datetime, 
                              duration: float, 
                              task_type: TaskType,
                              score: float) -> str:
        """Generate human-readable reasoning for slot selection"""
        hour = start_time.hour
        
        # Get pattern data
        preference = task_type.hourly_scores[hour]
        energy = task_type.performance_by_hour[hour] 
        confidence = task_type.confidence_scores[hour]
        cognitive_load = task_type.cognitive_load
        
        reasoning_parts = []
        
        # Time preference reasoning
        if preference > 0.7:
            reasoning_parts.append(f"High preference for {task_type.task_type} at {hour}:00")
        elif preference < 0.3:
            reasoning_parts.append(f"Low preference for {task_type.task_type} at {hour}:00")
        
        # Energy reasoning
        if energy > 0.7:
            reasoning_parts.append("High energy period")
        elif energy < 0.4:
            reasoning_parts.append("Low energy period")
        
        # Cognitive load matching
        if cognitive_load > 0.7 and energy > 0.6:
            reasoning_parts.append("Good energy for demanding task")
        elif cognitive_load > 0.7 and energy < 0.5:
            reasoning_parts.append("⚠️ High-cognitive task during low energy")
        
        # Confidence level
        if confidence > 0.8:
            reasoning_parts.append("High confidence in pattern")
        elif confidence < 0.3:
            reasoning_parts.append("Limited historical data")
        
        return " • ".join(reasoning_parts) if reasoning_parts else f"Score: {score:.2f}"
    
    async def find_optimal_slot(self, user_id: str, 
                              task_type: TaskType,
                              duration: float,
                              date_range: Tuple[datetime, datetime],
                              existing_events: List[Dict]) -> Optional[Dict]:
        """Find optimal time slot using comprehensive Tier 2 pattern analysis"""
        
        start_date, end_date = date_range
        slot_candidates = []
        
        # Try each potential start time (30-min intervals for speed)
        current_time = start_date.replace(minute=0, second=0, microsecond=0)
        
        while current_time + timedelta(hours=duration) <= end_date:
            # Check basic availability
            if self.is_slot_available(current_time, duration, existing_events):
                
                # Calculate comprehensive score using all Tier 2 patterns
                score = self.calculate_slot_score(
                    current_time, 
                    duration, 
                    task_type,
                    existing_events
                )
                
                # Store candidate with detailed scoring
                slot_candidates.append({
                    'start_time': current_time,
                    'end_time': current_time + timedelta(hours=duration),
                    'score': score,
                    'reasoning': self.generate_slot_reasoning(
                        current_time, duration, task_type, score
                    )
                })
            
            # Move to next 30-minute interval
            current_time += timedelta(minutes=30)
        
        if not slot_candidates:
            return None
        
        # Sort by score and return best option with reasoning
        best_slots = sorted(slot_candidates, key=lambda x: x['score'], reverse=True)
        
        return {
            'optimal_slot': best_slots[0],
            'alternatives': best_slots[1:3],  # Top 3 alternatives
            'pattern_insights': self.get_pattern_insights(task_type),
            'schedule_optimization': self.analyze_schedule_fit(
                best_slots[0], existing_events, task_type
            )
        }
    
    def get_pattern_insights(self, task_type: TaskType) -> Dict:
        """Extract key insights from Tier 2 patterns"""
        hourly_scores = task_type.hourly_scores
        performance_by_hour = task_type.performance_by_hour
        
        # Find peak hours
        peak_hours = [
            i for i, score in enumerate(hourly_scores) 
            if score > 0.7
        ]
        
        # Find energy peak hours
        energy_peaks = [
            i for i, energy in enumerate(performance_by_hour)
            if energy > 0.7
        ]
        
        # Find overlapping high-performance times
        optimal_hours = list(set(peak_hours) & set(energy_peaks))
        
        return {
            'best_hours': optimal_hours,
            'peak_preference_hours': peak_hours,
            'peak_energy_hours': energy_peaks,
            'cognitive_load': task_type.cognitive_load,
            'typical_duration': task_type.typical_duration,
            'recovery_needed': task_type.recovery_hours
        }
    
    def analyze_schedule_fit(self, selected_slot: Dict, 
                           existing_events: List[Dict],
                           task_type: TaskType) -> Dict:
        """Analyze how this slot fits within existing schedule"""
        start_time = selected_slot['start_time']
        
        # Find surrounding events
        before_events = [
            e for e in existing_events 
            if e['scheduled_end'] <= start_time
        ]
        after_events = [
            e for e in existing_events
            if e['scheduled_start'] >= selected_slot['end_time']
        ]
        
        analysis = {
            'buffer_before': None,
            'buffer_after': None,
            'day_load': 0,
            'context_switch_penalty': 0
        }
        
        # Calculate buffers
        if before_events:
            last_event = max(before_events, key=lambda e: e['scheduled_end'])
            buffer_minutes = (start_time - last_event['scheduled_end']).total_seconds() / 60
            analysis['buffer_before'] = buffer_minutes
        
        if after_events:
            next_event = min(after_events, key=lambda e: e['scheduled_start'])
            buffer_minutes = (next_event['scheduled_start'] - selected_slot['end_time']).total_seconds() / 60
            analysis['buffer_after'] = buffer_minutes
        
        # Calculate day cognitive load
        same_day_events = [
            e for e in existing_events
            if e['scheduled_start'].date() == start_time.date()
        ]
        analysis['day_load'] = len(same_day_events)
        
        return analysis
    
    async def schedule_event(self, user_id: str, 
                           request: ScheduleEventRequest) -> Dict:
        """Main scheduling function using Tier 2 patterns"""
        
        # 1. Find or create task type (Tier 2)
        similar_task = await self.task_type_service.find_similar_task_type(
            user_id, f"{request.title} {request.description or ''}"
        )
        
        if similar_task and similar_task.similarity > 0.8:
            print(f"📊 Found similar task: '{similar_task.task_type.task_type}' (similarity: {similar_task.similarity:.2f})")
            task_type = similar_task.task_type
        else:
            # Create new task type
            category = self.task_type_service.categorize_task_description(
                f"{request.title} {request.description or ''}"
            )
            print(f"🆕 Creating new task type: '{request.title}' (category: {category.value})")
            task_type = await self.task_type_service.create_task_type(
                user_id, request.title, category, request.description
            )
        
        # 2. Set up scheduling window
        start_date = request.preferred_date or datetime.now()
        if start_date < datetime.now():
            start_date = datetime.now()
            
        # Search next 7 days for optimal slot
        end_date = start_date + timedelta(days=7)
        
        # Mock existing events (in real implementation, get from database)
        existing_events = []
        
        # 3. Find optimal slot using patterns
        optimal_result = await self.find_optimal_slot(
            user_id,
            task_type,
            request.duration,
            (start_date, end_date),
            existing_events
        )
        
        if not optimal_result:
            raise ValueError("No available time slots found in the next 7 days")
        
        # 4. Create event data
        optimal_slot = optimal_result['optimal_slot']
        
        event_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "task_type_id": str(task_type.id),
            "title": request.title,
            "description": request.description,
            "scheduled_start": optimal_slot['start_time'],
            "scheduled_end": optimal_slot['end_time'],
            "deadline": request.deadline,
            "calculated_priority": 0.5,  # Would be calculated based on importance + urgency
            "created_at": datetime.now()
        }
        
        return {
            "event": event_data,
            "optimal_slot": optimal_slot,
            "alternatives": optimal_result['alternatives'],
            "pattern_insights": optimal_result['pattern_insights'],
            "schedule_optimization": optimal_result['schedule_optimization'],
            "task_type_used": {
                "id": str(task_type.id),
                "name": task_type.task_type,
                "category": task_type.category.value,
                "is_new": similar_task is None
            }
        } 