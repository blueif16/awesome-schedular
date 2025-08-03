"""
Scheduler Service - Core Scheduling Algorithm
Uses Tier 2 patterns to find optimal time slots
"""

import uuid
import math
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from models import TaskType, Event, ScheduleEventRequest, SchedulingResult
from task_type_service import TaskTypeService


class SchedulerService:
    def __init__(self, task_type_service: TaskTypeService):
        self.task_type_service = task_type_service
    
    def _get_next_clean_hour(self, dt: datetime) -> datetime:
        """Get next clean hour boundary from given datetime"""
        if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0:
            return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            return dt.replace(minute=0, second=0, microsecond=0)
    
    def calculate_slot_score(self, start_time: datetime, 
                           duration: float, 
                           task_type: TaskType,
                           existing_events: List[Dict] = None) -> float:
        """Calculate comprehensive score for a time slot using Tier 2 patterns"""
        
        weekly_habit_scores = task_type.weekly_habit_scores  # 168-hour weekly patterns
        slot_confidence = task_type.slot_confidence          # 7x24 confidence matrix
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
            # Convert Python weekday (0=Monday) to our schema (0=Sunday)
            python_weekday = start_time.weekday()  # 0=Monday, 6=Sunday
            day_of_week = (python_weekday + 1) % 7  # Convert to 0=Sunday, 6=Saturday
            
            # Convert to weekly index (day_of_week * 24 + hour)
            weekly_index = day_of_week * 24 + hour_index
            
            # Get pattern scores from Tier 2 arrays
            if weekly_index < len(weekly_habit_scores):
                preference = weekly_habit_scores[weekly_index]  # Weekly habit pattern
            else:
                preference = 0.5  # Neutral fallback
            
            # Get confidence from 7x24 matrix
            if (day_of_week < len(slot_confidence) and 
                hour_index < len(slot_confidence[day_of_week])):
                confidence = slot_confidence[day_of_week][hour_index]
            else:
                confidence = 0.1  # Low confidence fallback
            
            # Use preference as both preference and energy (simplified)
            base_score = preference
            weighted_score = base_score * max(0.1, confidence) * hour_fraction
            
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
        day_of_week = start_time.weekday()  # 0=Monday, 6=Sunday
        weekly_index = day_of_week * 24 + hour
        
        # Get pattern data from weekly arrays
        if weekly_index < len(task_type.weekly_habit_scores):
            preference = task_type.weekly_habit_scores[weekly_index]
        else:
            preference = 0.5  # Neutral fallback
            
        # Get confidence from 7x24 matrix
        if (day_of_week < len(task_type.slot_confidence) and 
            hour < len(task_type.slot_confidence[day_of_week])):
            confidence = task_type.slot_confidence[day_of_week][hour]
        else:
            confidence = 0.1  # Low confidence fallback
            
        cognitive_load = task_type.cognitive_load
        
        reasoning_parts = []
        
        # Time preference reasoning
        if preference > 0.7:
            reasoning_parts.append(f"High preference for {task_type.task_type} at {hour}:00")
        elif preference < 0.3:
            reasoning_parts.append(f"Low preference for {task_type.task_type} at {hour}:00")
        
        # Cognitive load reasoning
        if cognitive_load > 0.7:
            reasoning_parts.append("High-cognitive task")
        elif cognitive_load < 0.3:
            reasoning_parts.append("Low-cognitive task")
        
        # Confidence level
        if confidence > 0.8:
            reasoning_parts.append("High confidence in pattern")
        elif confidence < 0.3:
            reasoning_parts.append("Limited historical data")
        
        return " • ".join(reasoning_parts) if reasoning_parts else f"Score: {score:.2f}"
    
    async def find_optimal_slot(self, user_id: str, 
                              task_type: TaskType,
                              duration: float,
                              available_periods: List[Tuple[datetime, datetime]],
                              existing_events: List[Dict]) -> Optional[Dict]:
        """Find optimal time slot across multiple available time periods using Tier 2 patterns"""
        
        slot_candidates = []
        
        # Search within each available time period
        for period_start, period_end in available_periods:
            # Try each potential start time (30-min intervals for speed)
            current_time = period_start.replace(minute=0, second=0, microsecond=0)
            
            # Ensure we start at the beginning of the period or later
            if current_time < period_start:
                current_time = period_start
            
            while current_time + timedelta(hours=duration) <= period_end:
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
                        ),
                        'period': f"{period_start.strftime('%m/%d %H:%M')}-{period_end.strftime('%H:%M')}"
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
            ),
            'periods_searched': len(available_periods),
            'total_candidates': len(slot_candidates)
        }
    
    def get_pattern_insights(self, task_type: TaskType) -> Dict:
        """Extract key insights from Tier 2 patterns"""
        weekly_habit_scores = task_type.weekly_habit_scores
        
        # Find peak hours across the week (convert weekly index back to hours)
        peak_hours = []
        if weekly_habit_scores:
            for i, score in enumerate(weekly_habit_scores):
                if score > 0.7:
                    # Convert weekly index back to hour (0-23)
                    hour = i % 24
                    if hour not in peak_hours:
                        peak_hours.append(hour)
        
        return {
            'best_hours': peak_hours,
            'cognitive_load': task_type.cognitive_load,
            'typical_duration': task_type.typical_duration,
            'recovery_needed': task_type.recovery_hours,
            'total_completions': task_type.completion_count
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
                           request: ScheduleEventRequest,
                           available_periods: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict:
        """Main scheduling function using Tier 2 patterns
        
        Args:
            user_id: User identifier
            request: Scheduling request details
            available_periods: List of [start, end] time periods when scheduling is allowed.
                             If None, defaults to next 7 days from preferred_date
        """
        
        # 1. Find or create task type (Tier 2)
        print(f"🔍 SCHEDULER: Looking for similar task type...")
        print(f"   📝 Query: '{request.title} {request.description or ''}'")
        print(f"   👤 User ID: {user_id[:8]}...")
        
        similar_task = await self.task_type_service.find_similar_task_type(
            user_id, f"{request.title} {request.description or ''}"
        )
        
        if similar_task and similar_task.similarity > 0.4:
            print(f"🎯 USING EXISTING TASK TYPE: '{similar_task.task_type.task_type}' (completions: {similar_task.task_type.completion_count})")
            task_type = similar_task.task_type
        else:
            # Create new task type with robust error handling
            if similar_task:
                print(f"🔄 Similarity {similar_task.similarity:.3f} < 0.4 threshold - creating new task type")
            else:
                print(f"🆕 No similar task types found - creating new task type")
            print(f"🚀 SCHEDULER: Calling TaskTypeService.create_task_type('{request.title}')")
            try:
                task_type = await self.task_type_service.create_task_type(
                    user_id, request.title, request.description, self
                )
                print(f"✅ SCHEDULER: Task type created successfully: {task_type.task_type}")
            except Exception as create_error:
                print(f"❌ SCHEDULER: Task type creation failed: {create_error}")
                if "duplicate key" in str(create_error).lower():
                    # Handle race condition - task was created by another process
                    print(f"⚠️ SCHEDULER: Race condition detected - task type '{request.title}' was created by another process")
                    print(f"🔄 SCHEDULER: Attempting to fetch existing task type...")
                    similar_task_retry = await self.task_type_service.find_similar_task_type(
                        user_id, f"{request.title} {request.description or ''}"
                    )
                    if similar_task_retry and similar_task_retry.similarity > 0.7:
                        task_type = similar_task_retry.task_type
                        print(f"🔗 SCHEDULER: Successfully found existing task type: {task_type.task_type}")
                    else:
                        print(f"❌ SCHEDULER: Could not find suitable existing task type")
                        raise create_error  # Re-raise if we can't find a good match
                else:
                    print(f"❌ SCHEDULER: Unhandled task type creation error")
                    raise create_error
        
        # 2. Set up available time periods
        if available_periods is None:
            # Default: search next 7 days from preferred date, starting at next clean hour
            raw_start = request.preferred_date or datetime.now()
            if raw_start < datetime.now():
                raw_start = datetime.now()
            
            # Round up to next clean hour boundary
            start_date = self._get_next_clean_hour(raw_start)
            
            end_date = start_date + timedelta(days=7)
            available_periods = [(start_date, end_date)]
            print(f"📅 Using default 7-day window: {start_date.strftime('%m/%d %H:%M')} - {end_date.strftime('%m/%d %H:%M')}")
        else:
            periods_str = ", ".join([
                f"{start.strftime('%m/%d %H:%M')}-{end.strftime('%H:%M')}" 
                for start, end in available_periods
            ])
            print(f"📅 Searching in {len(available_periods)} available periods: {periods_str}")
        
        # Mock existing events (in real implementation, get from database)
        existing_events = []
        
        # 3. Find optimal slot using patterns across available periods
        optimal_result = await self.find_optimal_slot(
            user_id,
            task_type,
            request.duration,
            available_periods,
            existing_events
        )
        
        if not optimal_result:
            raise ValueError(f"No available time slots found in {len(available_periods)} time periods")
        
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
            "calculated_priority": optimal_slot['score']
        }
        
        return {
            "event": event_data,
            "optimal_slot": optimal_slot,
            "alternatives": optimal_result.get('alternatives', []),
            "pattern_insights": optimal_result.get('pattern_insights', {}),
            "task_type_used": {
                "id": str(task_type.id),
                "name": task_type.task_type
            },
            "scheduling_stats": {
                "periods_searched": optimal_result.get('periods_searched', 0),
                "total_candidates": optimal_result.get('total_candidates', 0)
            }
        } 
    
    async def schedule_with_behavioral_patterns(self, 
                                          user_id: str,
                                          user_energy_pattern: List[float], 
                                          request: ScheduleEventRequest,
                                          existing_events: List[Dict],
                                          available_periods: Optional[List[Tuple[datetime, datetime]]] = None,
                                          openai_client = None,
                                          memory_service = None) -> Dict:
        """
        Enhanced behavioral scheduling using multiple pattern factors:
        - Task habit patterns (when user typically does this task)
        - User energy patterns (when user has highest/lowest energy)
        - Cognitive load matching (high-cognitive tasks during high-energy periods)
        - Duration optimization (respecting typical duration patterns)
        
        Uses RAG to find similar task_type, falls back to LLM if no good match
        """
        
        # 1. Find similar task type via RAG/similarity search
        similar_task = await self.task_type_service.find_similar_task_type(
            user_id, f"{request.title} {request.description or ''}"
        )
        
        if similar_task and similar_task.similarity > 0.4:
            task_type = similar_task.task_type
            print(f"🎯 BEHAVIORAL SCHEDULING: Using patterns from '{task_type.task_type}' (completions: {task_type.completion_count})")
            
            # 2. Set up time periods at clean hour boundaries
            if available_periods is None:
                raw_start = request.preferred_date or datetime.now()
                if raw_start < datetime.now():
                    raw_start = datetime.now()
                
                # Round up to next clean hour boundary
                start_date = self._get_next_clean_hour(raw_start)
                
                end_date = start_date + timedelta(days=7)
                available_periods = [(start_date, end_date)]
            
            # 3. Enhanced behavioral scoring considering all patterns
            optimal_result = await self.find_optimal_slot_with_energy(
                user_id, task_type, user_energy_pattern, request.duration, available_periods, existing_events
            )
            
            if not optimal_result:
                raise ValueError("No available time slots found using enhanced behavioral patterns")
            
            # 4. Create event with comprehensive behavioral scoring
            optimal_slot = optimal_result['optimal_slot']
            
            return {
                "event": {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "task_type_id": str(task_type.id),
                    "title": request.title,
                    "description": request.description,
                    "scheduled_start": optimal_slot['start_time'],
                    "scheduled_end": optimal_slot['end_time'],
                    "calculated_priority": optimal_slot['score']
                },
                "scheduling_method": "enhanced_behavioral_patterns",
                "scoring_factors": {
                    "habit_patterns": True,
                    "energy_cognitive_fit": True
                },
                "optimal_slot": optimal_slot,
                "alternatives": optimal_result.get('alternatives', []),
                "pattern_insights": optimal_result.get('pattern_insights', {}),
                "task_type_used": {
                    "id": str(task_type.id),
                    "name": task_type.task_type,
                    "similarity_score": similar_task.similarity,
                    "cognitive_load": task_type.cognitive_load,
                    "typical_duration": task_type.typical_duration
                }
            }
        else:
            # Fallback to LLM scheduling when no good behavioral match found
            similarity_msg = f"similarity: {similar_task.similarity:.3f}" if similar_task else "no matches found"
            print(f"🔄 FALLBACK TO LLM: {similarity_msg} - insufficient for behavioral scheduling")
            
            # Generate user preferences context for new task type
            user_preferences = f"Creating new task type: '{request.title}'. " + \
                             f"Description: {request.description or 'No description provided'}. " + \
                             "Please analyze this task and suggest optimal scheduling patterns."
            
            return await self.schedule_with_llm(
                user_id=user_id,
                request=request,
                user_preferences=user_preferences,
                existing_events=existing_events,
                available_periods=available_periods,
                openai_client=openai_client,
                memory_service=memory_service
            )
    
    async def schedule_with_llm(self,
                              user_id: str,
                              request: ScheduleEventRequest,
                              user_preferences: str,
                              existing_events: List[Dict],
                              available_periods: Optional[List[Tuple[datetime, datetime]]] = None,
                              openai_client = None,
                              memory_service = None) -> Dict:
        """
        Active scheduling using LLM + behavioral patterns
        LLM interprets semantic preferences and calendar constraints
        """
        
        # 1. Find/create task type for behavioral context
        similar_task = await self.task_type_service.find_similar_task_type(
            user_id, f"{request.title} {request.description or ''}"
        )
        
        if similar_task and similar_task.similarity > 0.4:
            task_type = similar_task.task_type
            print(f"🎯 LLM SCHEDULING: Using existing task type '{task_type.task_type}' (completions: {task_type.completion_count})")
        else:
            if similar_task:
                print(f"🔄 Similarity {similar_task.similarity:.3f} < 0.4 threshold - creating new task type for LLM scheduling")
            print(f"🆕 Creating new task type for LLM: '{request.title}'")
            try:
                task_type = await self.task_type_service.create_task_type(
                    user_id, request.title, request.description, self
                )
            except Exception as create_error:
                if "duplicate key" in str(create_error).lower():
                    # Handle race condition - task was created by another process
                    print(f"⚠️ Task type '{request.title}' was created by another process, fetching existing...")
                    similar_task_retry = await self.task_type_service.find_similar_task_type(
                        user_id, f"{request.title} {request.description or ''}"
                    )
                    if similar_task_retry and similar_task_retry.similarity > 0.7:
                        task_type = similar_task_retry.task_type
                        print(f"🔗 RACE CONDITION RESOLVED: Using existing task type '{task_type.task_type}' (similarity: {similar_task_retry.similarity:.3f})")
                    else:
                        print(f"❌ Race condition but similarity too low: {similar_task_retry.similarity:.3f} < 0.7")
                        raise create_error  # Re-raise if we can't find a good match
                else:
                    raise create_error
        
        # 2. Set up time periods at clean hour boundaries
        if available_periods is None:
            raw_start = request.preferred_date or datetime.now()
            if raw_start < datetime.now():
                raw_start = datetime.now()
            
            # Round up to next clean hour boundary
            start_date = self._get_next_clean_hour(raw_start)
            
            end_date = start_date + timedelta(days=7)
            available_periods = [(start_date, end_date)]
        
        # 3. Find available free slots from calendar data
        free_slots = self._extract_free_slots_from_periods(
            available_periods, existing_events, request.duration
        )
        
        # 4. Query existing patterns from Mem0
        mem0_context = ""
        if memory_service:
            from mem0_service import Mem0Service
            mem0_service = Mem0Service(memory_service, self.task_type_service)
            mem0_context = await mem0_service.query_scheduling_context(
                user_id, task_type, request
            )
        
        # 5. Format context for LLM with behavioral patterns + calendar availability + mem0 context
        llm_context = self._prepare_llm_scheduling_context(
            task_type=task_type,
            user_preferences=user_preferences,
            free_slots=free_slots,
            existing_events=existing_events,
            request=request,
            mem0_context=mem0_context
        )
        
        # 6. Get LLM scheduling decision
        if not openai_client:
            raise ValueError("OpenAI client required for LLM scheduling")
        
        llm_response = await self._call_llm_for_scheduling(
            openai_client, llm_context, free_slots
        )
        
        # 7. Parse LLM response and create event
        selected_slot = self._parse_llm_scheduling_response(llm_response, free_slots)
        
        # 8. Extract and store new scheduling preferences from user input
        if memory_service and user_preferences:
            from mem0_service import Mem0Service
            mem0_service = Mem0Service(memory_service, self.task_type_service)
            await mem0_service.store_scheduling_preferences(
                user_id, user_preferences, task_type, llm_response.get('detected_patterns', []), openai_client
            )
        
        return {
            "event": {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "task_type_id": str(task_type.id),
                "title": request.title,
                "description": request.description,
                "scheduled_start": selected_slot['start_time'],
                "scheduled_end": selected_slot['end_time'],
                "calculated_priority": selected_slot.get('priority', 0.8)
            },
            "scheduling_method": "llm_semantic_analysis",
            "llm_reasoning": llm_response.get('reasoning', ''),
            "llm_confidence": llm_response.get('confidence', 0.8),
            "detected_patterns": llm_response.get('detected_patterns', []),
            "user_preferences_applied": user_preferences,
            "behavioral_context_used": {
                "task_type": task_type.task_type,
                "completion_count": task_type.completion_count,
                "cognitive_load": task_type.cognitive_load
            },
            "calendar_slots_considered": len(free_slots),
            "patterns_stored_in_mem0": memory_service is not None
        }
    
    def _extract_free_slots_from_periods(self, 
                                       available_periods: List[Tuple[datetime, datetime]],
                                       existing_events: List[Dict],
                                       duration: float) -> List[Dict]:
        """Extract actual free time slots that can fit the task duration"""
        free_slots = []
        
        for period_start, period_end in available_periods:
            current_time = period_start.replace(minute=0, second=0, microsecond=0)
            
            while current_time + timedelta(hours=duration) <= period_end:
                if self.is_slot_available(current_time, duration, existing_events):
                    free_slots.append({
                        'start_time': current_time,
                        'end_time': current_time + timedelta(hours=duration),
                        'day_of_week': current_time.strftime('%A'),
                        'time_of_day': current_time.strftime('%H:%M'),
                        'date': current_time.strftime('%Y-%m-%d')
                    })
                
                current_time += timedelta(minutes=30)
        
        return free_slots
    
    def _prepare_llm_scheduling_context(self,
                                      task_type: TaskType,
                                      user_preferences: str,
                                      free_slots: List[Dict],
                                      existing_events: List[Dict],
                                      request: ScheduleEventRequest,
                                      mem0_context: str = "") -> str:
        """Prepare comprehensive context for LLM scheduling decision"""
        
        # Get behavioral insights
        pattern_insights = self.get_pattern_insights(task_type)
        
        # Format calendar context
        free_slots_summary = []
        for slot in free_slots[:20]:  # Limit to prevent token overflow
            free_slots_summary.append(
                f"{slot['day_of_week']} {slot['date']} at {slot['time_of_day']}"
            )
        
        # Existing events context for the day
        today_events = [
            f"{e['title']} from {e['scheduled_start'].strftime('%H:%M')} to {e['scheduled_end'].strftime('%H:%M')}"
            for e in existing_events
            if e['scheduled_start'].date() == datetime.now().date()
        ]
        
        context = f"""
TASK TO SCHEDULE:
- Title: {request.title}
- Description: {request.description or 'None provided'}
- Duration: {request.duration} hours
- Cognitive Load: {task_type.cognitive_load:.2f} (0=easy, 1=demanding)

USER PREFERENCES (CURRENT REQUEST):
{user_preferences}

HISTORICAL PATTERNS FROM MEMORY:
{mem0_context if mem0_context else 'No previous patterns found'}

BEHAVIORAL PATTERNS FOR THIS TASK TYPE:
- Task Type: {task_type.task_type}
- Completion History: {task_type.completion_count} times
- Best Hours: {pattern_insights.get('best_hours', [])}
- Peak Energy Hours: {pattern_insights.get('peak_energy_hours', [])}
- Recovery Time Needed: {task_type.recovery_hours} hours

AVAILABLE FREE SLOTS:
{chr(10).join(free_slots_summary)}

TODAY'S EXISTING EVENTS:
{chr(10).join(today_events) if today_events else 'No events scheduled today'}

Please analyze the user preferences, historical patterns, behavioral data, and available slots to recommend the best scheduling option.
Consider energy levels, task complexity, user's stated preferences, and learned patterns from memory.
"""
        return context
    
    async def _call_llm_for_scheduling(self, openai_client, context: str, free_slots: List[Dict]) -> Dict:
        """Call LLM to make scheduling decision with structured output"""
        
        # Create structured schema for slot selection
        slot_options = []
        for i, slot in enumerate(free_slots[:10]):  # Limit to top 10 to prevent token overflow
            slot_options.append({
                "index": i,
                "description": f"{slot['day_of_week']} {slot['date']} at {slot['time_of_day']}"
            })
        
        function_schema = {
            "name": "select_optimal_slot",
            "description": "Select the optimal time slot for scheduling based on user preferences and behavioral patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_slot_index": {
                        "type": "integer",
                        "description": f"Index of selected slot from available options (0-{len(slot_options)-1})",
                        "minimum": 0,
                        "maximum": len(slot_options) - 1
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for why this slot was chosen, considering user preferences, behavioral patterns, and energy levels"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score in this scheduling decision (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "detected_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of scheduling patterns detected from user preferences (e.g., 'prefers evenings', 'needs quiet time', 'avoids mornings')"
                    }
                },
                "required": ["selected_slot_index", "reasoning", "confidence", "detected_patterns"]
            }
        }
        
        # Enhanced context with slot options
        enhanced_context = context + f"\n\nAVAILABLE SLOT OPTIONS:\n"
        for option in slot_options:
            enhanced_context += f"Option {option['index']}: {option['description']}\n"
        
        enhanced_context += "\nPlease select the best option and explain your reasoning."
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an intelligent scheduling assistant. Analyze the user's preferences, behavioral patterns, and available time slots to make optimal scheduling decisions. Focus on semantic understanding of preferences like 'evenings', 'after coffee', 'quiet time', etc."
                    },
                    {
                        "role": "user",
                        "content": enhanced_context
                    }
                ],
                functions=[function_schema],
                function_call={"name": "select_optimal_slot"},
                temperature=0.3
            )
            
            # Parse function call response
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "select_optimal_slot":
                result = json.loads(function_call.arguments)
                
                # Validate slot index
                selected_index = result.get("selected_slot_index", 0)
                if selected_index >= len(free_slots):
                    selected_index = 0  # Fallback to first slot
                
                return {
                    "selected_slot_index": selected_index,
                    "reasoning": result.get("reasoning", "LLM scheduling decision"),
                    "confidence": result.get("confidence", 0.8),
                    "detected_patterns": result.get("detected_patterns", []),
                    "raw_response": response.choices[0].message.content
                }
            else:
                raise ValueError("LLM did not return expected function call")
                
        except Exception as e:
            print(f"❌ Error calling LLM for scheduling: {e}")
            # Fallback to first available slot
            return {
                "selected_slot_index": 0,
                "reasoning": f"Fallback scheduling due to LLM error: {str(e)}",
                "confidence": 0.5,
                "detected_patterns": [],
                "error": str(e)
            }
    
    def _parse_llm_scheduling_response(self, llm_response: Dict, free_slots: List[Dict]) -> Dict:
        """Parse LLM response and return selected time slot"""
        selected_index = llm_response.get('selected_slot_index', 0)
        if selected_index < len(free_slots):
            selected_slot = free_slots[selected_index].copy()
            # Add LLM metadata to the selected slot
            selected_slot['priority'] = llm_response.get('confidence', 0.8)
            selected_slot['llm_reasoning'] = llm_response.get('reasoning', '')
            return selected_slot
        else:
            # Fallback to first slot if index is invalid
            if free_slots:
                fallback_slot = free_slots[0].copy()
                fallback_slot['priority'] = 0.5
                fallback_slot['llm_reasoning'] = 'Fallback due to invalid slot selection'
                return fallback_slot
            else:
                return None
    
    async def parse_and_apply_time_preferences(self, 
                                          user_id: str,
                                          task_type: TaskType, 
                                          user_preferences: str,
                                          openai_client) -> bool:
        """Parse user time preferences and apply to task type behavioral patterns using LLM"""
        
        function_schema = {
            "name": "extract_time_preferences",
            "description": "Extract time preferences as compact string",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_patterns": {
                        "type": "string",
                        "description": "Compact time patterns: 'days:hour_start-hour_end:boost,days:hour_start-hour_end:boost'. Days: 0-6=Mon-Sun, 0-4=weekdays, 5-6=weekend. Example: '0-6:6-11:0.8,5-6:8-12:0.9'"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["time_patterns", "confidence"]
            }
        }
        
        context = f"""
Extract time preferences as compact string format:

User Input: "{user_preferences}"
Task Type: "{task_type.task_type}"

Output format: "days:start-end:boost,days:start-end:boost"
- Days: 0-6 (Mon-Sun), 0-4 (weekdays), 5-6 (weekend), or specific like 0,2,4
- Hours: 0-23 
- Boost: 0.0-1.0

Examples:
- "morning" → "0-6:6-11:0.8"
- "weekends evening" → "5-6:17-21:0.9"  
- "Monday afternoon" → "0:12-16:0.8"
- "not morning" → "0-6:6-11:0.2"

Be concise! Output only the pattern string.
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract specific time preferences into day/hour/boost format. Be precise and concise."},
                    {"role": "user", "content": context}
                ],
                functions=[function_schema],
                function_call={"name": "extract_time_preferences"},
                temperature=0.2
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "extract_time_preferences":
                result = json.loads(function_call.arguments)
                time_patterns = result.get("time_patterns", "")
                confidence = result.get("confidence", 0.0)
                
                if not time_patterns or confidence < 0.3:
                    print(f"⚠️ Low confidence ({confidence:.2f}) in time preference extraction")
                    return False
                
                # Parse the compact string format
                parsed_patterns = self._parse_time_pattern_string(time_patterns)
                if not parsed_patterns:
                    print(f"⚠️ Could not parse time patterns: '{time_patterns}'")
                    return False
                
                print(f"🎯 Extracted {len(parsed_patterns)} time patterns (confidence: {confidence:.2f})")
                print(f"📋 Pattern string: '{time_patterns}'")
                
                # Apply to behavioral arrays
                updated = self._apply_time_patterns_to_task_type(task_type, parsed_patterns)
                if updated:
                    # Save to database
                    success = await self._save_updated_task_type_patterns(task_type)
                    if success:
                        # Show what was applied
                        print(f"✅ Applied time preferences to '{task_type.task_type}':")
                        for pattern in parsed_patterns:
                            days_str = ",".join(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d] for d in pattern["days"])
                            print(f"   {days_str} {pattern['hour_start']:02d}:00-{pattern['hour_end']:02d}:00 → boost {pattern['boost']:.1f}")
                        return True
                    else:
                        print(f"⚠️ Failed to save patterns to database")
                        return False
                else:
                    print(f"⚠️ No valid time patterns to apply")
                    return False
            else:
                print(f"❌ LLM did not return expected function call")
                return False
                
        except Exception as e:
            print(f"❌ Error parsing time preferences: {e}")
            return False
    
    def _apply_time_patterns_to_task_type(self, task_type: TaskType, time_patterns: List[Dict]) -> bool:
        """Apply parsed time patterns to task type behavioral arrays"""
        if not time_patterns:
            return False
            
        # Work with copies to avoid modifying original
        weekly_habit_scores = task_type.weekly_habit_scores.copy()
        slot_confidence = [row.copy() for row in task_type.slot_confidence]
        
        applied_count = 0
        for pattern in time_patterns:
            days = pattern.get("days", [])
            hour_start = pattern.get("hour_start")
            hour_end = pattern.get("hour_end") 
            boost = pattern.get("boost", 0.5)
            
            # Validate pattern
            if (not days or hour_start is None or hour_end is None or 
                not all(0 <= d <= 6 for d in days) or 
                not (0 <= hour_start <= 23) or not (0 <= hour_end <= 23)):
                continue
                
            # Apply to all day/hour combinations in the range
            for day in days:
                for hour in range(hour_start, hour_end + 1):  # inclusive range
                    # Apply to weekly habit scores (168-element array)
                    weekly_index = day * 24 + hour
                    if weekly_index < len(weekly_habit_scores):
                        weekly_habit_scores[weekly_index] = boost
                        applied_count += 1
                    
                    # Apply to slot confidence matrix (7x24)
                    if day < len(slot_confidence) and hour < len(slot_confidence[day]):
                        # Use boost value as confidence (higher preference = higher confidence)
                        slot_confidence[day][hour] = min(0.9, boost * 1.1)  # Scale slightly for confidence
        
        if applied_count > 0:
            # Update the task type object
            task_type.weekly_habit_scores = weekly_habit_scores
            task_type.slot_confidence = slot_confidence
            print(f"🔄 Updated {applied_count} time slots from {len(time_patterns)} patterns")
            return True
        else:
            print(f"⚠️ No valid time slots found to apply")
            return False
    
    async def _save_updated_task_type_patterns(self, task_type: TaskType) -> bool:
        """Save updated behavioral patterns to database"""
        try:
            from task_type_service import TaskTypeService
            
            # Use the task type service to update patterns
            result = self.task_type_service.supabase.table("task_types").update({
                "weekly_habit_scores": task_type.weekly_habit_scores,
                "slot_confidence": task_type.slot_confidence,
                "updated_at": datetime.now().isoformat()
            }).eq("id", str(task_type.id)).execute()
            
            if result.data:
                return True
            else:
                print(f"⚠️ Failed to update database - no data returned")
                return False
                
        except Exception as e:
            print(f"❌ Error saving behavioral patterns: {e}")
            return False 

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
    
    async def analyze_task_characteristics(self, openai_client, task_type: str, description: str = None) -> Dict[str, float]:
        """Use LLM to analyze task characteristics: cognitive load, importance, duration, and recovery time"""
        
        # Prepare task context for analysis
        task_context = f"Task: {task_type}"
        if description:
            task_context += f"\nDescription: {description}"
        
        function_schema = {
            "name": "analyze_task_characteristics",
            "description": "Analyze task characteristics",
            "parameters": {
                "type": "object",
                "properties": {
                    "cognitive_load": {
                        "type": "number",
                        "description": "Mental effort (0.0-1.0): 0.2=routine, 0.5=moderate, 0.8=complex",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "importance_score": {
                        "type": "number", 
                        "description": "Task importance (0.0-1.0): 0.3=low, 0.5=normal, 0.8=critical",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "typical_duration": {
                        "type": "number",
                        "description": "Duration in hours (0.25-8.0)",
                        "minimum": 0.25,
                        "maximum": 8.0
                    },
                    "recovery_hours": {
                        "type": "number",
                        "description": "Recovery time (0.0-2.0)",
                        "minimum": 0.0,
                        "maximum": 2.0
                    }
                },
                "required": ["cognitive_load", "importance_score"]
            }
        }
        
        analysis_prompt = f"""
Analyze: {task_context}

Examples:
- "Email": cognitive_load=0.2, importance=0.4
- "Strategic planning": cognitive_load=0.8, importance=0.8  
- "Code review": cognitive_load=0.7, importance=0.6
- "Social media": cognitive_load=0.2, importance=0.3
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze task cognitive load (mental effort) and importance. Be precise."
                    },
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ],
                functions=[function_schema],
                function_call={"name": "analyze_task_characteristics"},
                temperature=0.3
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "analyze_task_characteristics":
                result = json.loads(function_call.arguments)
                
                # Validate and clamp values with smart defaults
                cognitive_load = max(0.0, min(1.0, result.get("cognitive_load", 0.5)))
                importance_score = max(0.0, min(1.0, result.get("importance_score", 0.5)))
                
                # Smart defaults based on cognitive load if not provided
                typical_duration = result.get("typical_duration")
                if typical_duration is None:
                    # Higher cognitive load = longer typical duration
                    typical_duration = 0.5 + (cognitive_load * 2.0)  # 0.5-2.5 hours
                typical_duration = max(0.25, min(8.0, typical_duration))
                
                recovery_hours = result.get("recovery_hours")
                if recovery_hours is None:
                    # Recovery scales with cognitive load
                    recovery_hours = cognitive_load * 0.6  # 0.0-0.6 hours
                recovery_hours = max(0.0, min(2.0, recovery_hours))
                
                analysis = {
                    "cognitive_load": cognitive_load,
                    "importance_score": importance_score,
                    "typical_duration": typical_duration,
                    "recovery_hours": recovery_hours,
                    "reasoning": f"LLM analysis: cognitive={cognitive_load:.1f}, importance={importance_score:.1f}"
                }
                
                print(f"   💭 LLM Result: {analysis['reasoning']}")
                return analysis
            else:
                raise ValueError("LLM did not return expected function call")
                
        except Exception as e:
            print(f"   ❌ Error in LLM task analysis: {e}")
            # Return intelligent defaults based on task keywords
            return self._get_fallback_task_analysis(task_type, description)
    
    def _get_fallback_task_analysis(self, task_type: str, description: str = None) -> Dict[str, float]:
        """Provide intelligent fallback analysis based on task keywords"""
        
        task_text = f"{task_type} {description or ''}".lower()
        
        # Default values
        cognitive_load = 0.5
        importance_score = 0.5
        typical_duration = 1.0
        recovery_hours = 0.3
        
        # Keyword-based adjustments
        high_cognitive_keywords = ['analysis', 'planning', 'design', 'programming', 'research', 'strategy', 'complex', 'technical', 'creative', 'presentation']
        low_cognitive_keywords = ['email', 'check', 'review', 'update', 'routine', 'admin', 'simple', 'quick']
        
        high_importance_keywords = ['critical', 'urgent', 'client', 'meeting', 'ceo', 'presentation', 'deadline', 'important', 'emergency']
        low_importance_keywords = ['optional', 'later', 'when possible', 'free time', 'break', 'casual']
        
        long_duration_keywords = ['workshop', 'training', 'session', 'deep', 'comprehensive', 'full', 'complete']
        short_duration_keywords = ['quick', 'brief', 'check', 'update', 'short', 'scan']
        
        # Adjust cognitive load
        if any(keyword in task_text for keyword in high_cognitive_keywords):
            cognitive_load = 0.7
        elif any(keyword in task_text for keyword in low_cognitive_keywords):
            cognitive_load = 0.3
            
        # Adjust importance
        if any(keyword in task_text for keyword in high_importance_keywords):
            importance_score = 0.8
        elif any(keyword in task_text for keyword in low_importance_keywords):
            importance_score = 0.3
            
        # Adjust duration
        if any(keyword in task_text for keyword in long_duration_keywords):
            typical_duration = 2.5
        elif any(keyword in task_text for keyword in short_duration_keywords):
            typical_duration = 0.5
            
        # Adjust recovery based on cognitive load
        recovery_hours = cognitive_load * 0.5  # Higher cognitive load = more recovery
        
        print(f"   🔄 Using keyword-based fallback analysis")
        return {
            "cognitive_load": cognitive_load,
            "importance_score": importance_score,
            "typical_duration": typical_duration,
            "recovery_hours": recovery_hours,
            "reasoning": "Keyword-based fallback analysis"
        }

    async def find_optimal_slot_with_energy(self, user_id: str, 
                                           task_type: TaskType,
                                           user_energy_pattern: List[float],
                                           duration: float,
                                           available_periods: List[Tuple[datetime, datetime]],
                                           existing_events: List[Dict]) -> Optional[Dict]:
        """Enhanced slot finding using comprehensive behavioral patterns:
        - Task habit patterns (when user typically does this task)
        - User energy patterns (when user has highest/lowest energy) 
        - Cognitive load matching (high-cognitive tasks during high-energy periods)
        - Duration optimization (respecting typical duration vs requested duration)
        """
        
        slot_candidates = []
        
        # Search within each available time period
        for period_start, period_end in available_periods:
            # Try each potential start time (30-min intervals for speed)
            current_time = period_start.replace(minute=0, second=0, microsecond=0)
            
            # Ensure we start at the beginning of the period or later
            if current_time < period_start:
                current_time = period_start
            
            while current_time + timedelta(hours=duration) <= period_end:
                # Check basic availability
                if self.is_slot_available(current_time, duration, existing_events):
                    
                    # Calculate comprehensive score using all behavioral factors
                    score = self.calculate_enhanced_slot_score(
                        current_time, 
                        duration, 
                        task_type,
                        user_energy_pattern,
                        existing_events
                    )
                    
                    # Store candidate with detailed scoring
                    slot_candidates.append({
                        'start_time': current_time,
                        'end_time': current_time + timedelta(hours=duration),
                        'score': score,
                        'reasoning': self.generate_enhanced_slot_reasoning(
                            current_time, duration, task_type, user_energy_pattern, score
                        ),
                        'period': f"{period_start.strftime('%m/%d %H:%M')}-{period_end.strftime('%H:%M')}"
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
            'pattern_insights': self.get_enhanced_pattern_insights(task_type, user_energy_pattern),
            'schedule_optimization': self.analyze_schedule_fit(
                best_slots[0], existing_events, task_type
            ),
            'periods_searched': len(available_periods),
            'total_candidates': len(slot_candidates)
        }

    def calculate_enhanced_slot_score(self, start_time: datetime, 
                           duration: float, 
                           task_type: TaskType,
                           user_energy_pattern: List[float],
                           existing_events: List[Dict] = None) -> float:
        """Calculate comprehensive score for a time slot using enhanced behavioral patterns"""
        
        # Get patterns from task_type (Tier 2)
        weekly_habit_scores = task_type.weekly_habit_scores  # 168-hour weekly patterns
        slot_confidence = task_type.slot_confidence          # 7x24 confidence matrix
        cognitive_load = task_type.cognitive_load
        recovery_hours = task_type.recovery_hours
        
        # Get user energy patterns (Tier 3)
        user_energy_scores = user_energy_pattern
        
        total_score = 0
        total_weight = 0
        
        # Score each hour the task spans
        current_hour = start_time.hour
        remaining_duration = duration
        hour_scores = []
        
        while remaining_duration > 0:
            hour_fraction = min(1.0, remaining_duration)
            hour_index = current_hour % 24
            # Convert Python weekday (0=Monday) to our schema (0=Sunday)
            python_weekday = start_time.weekday()  # 0=Monday, 6=Sunday
            day_of_week = (python_weekday + 1) % 7  # Convert to 0=Sunday, 6=Saturday
            
            # Convert to weekly index (day_of_week * 24 + hour)
            weekly_index = day_of_week * 24 + hour_index
            
            # Get pattern scores from Tier 2 arrays
            if weekly_index < len(weekly_habit_scores):
                preference = weekly_habit_scores[weekly_index]  # Weekly habit pattern
            else:
                preference = 0.5  # Neutral fallback
            
            # Get confidence from 7x24 matrix
            if (day_of_week < len(slot_confidence) and 
                hour_index < len(slot_confidence[day_of_week])):
                confidence = slot_confidence[day_of_week][hour_index]
            else:
                confidence = 0.1  # Low confidence fallback
            
            # Get user energy score at this weekly time
            if weekly_index < len(user_energy_scores):
                energy_score = user_energy_scores[weekly_index]
            else:
                energy_score = 0.5  # Neutral fallback
            
            # Enhanced scoring combining all factors:
            # 1. Task habit preference (50% weight)
            # 2. Energy-cognitive fit (50% weight) - energy adjusted for cognitive load requirements
            habit_component = preference * 0.5
            
            # Energy-cognitive fit: adjust energy score based on task's cognitive demands
            if cognitive_load > 0.7:  # High cognitive load task
                # High-cognitive tasks need high energy - penalize low energy periods
                energy_cognitive_fit = energy_score  # Direct energy score (0.0-1.0)
            elif cognitive_load < 0.3:  # Low cognitive load task  
                # Low-cognitive tasks can work with lower energy - boost low energy periods
                energy_cognitive_fit = 1.0 - (1.0 - energy_score) * 0.5  # Reduce penalty for low energy
            else:  # Medium cognitive load
                # Medium-cognitive tasks have moderate energy requirements
                energy_cognitive_fit = energy_score * 0.8 + 0.2  # Slight boost across all energy levels
            
            energy_component = energy_cognitive_fit * 0.5
            
            base_score = habit_component + energy_component
            weighted_score = base_score * max(0.1, confidence) * hour_fraction
            
            hour_scores.append({
                'hour': current_hour,
                'base_score': base_score,
                'confidence': confidence,
                'energy_score': energy_score,
                'habit_preference': preference,
                'energy_cognitive_fit': energy_cognitive_fit,
                'weighted_score': weighted_score
            })
            
            total_score += weighted_score
            total_weight += confidence * hour_fraction
            
            remaining_duration -= hour_fraction
            current_hour += 1
        
        # Base score from patterns
        pattern_score = total_score / total_weight if total_weight > 0 else 0.5
        

        
        # Apply additional cognitive load penalty if average energy is too low
        avg_energy = sum(h['energy_score'] for h in hour_scores) / len(hour_scores)
        if cognitive_load > 0.7 and avg_energy < 0.4:
            pattern_score *= 0.6  # Strong penalty for high-cognitive tasks during very low energy
        elif cognitive_load > 0.5 and avg_energy < 0.3:
            pattern_score *= 0.7  # Moderate penalty
        
        # Check recovery time from previous tasks
        recovery_penalty = self.calculate_recovery_penalty(
            start_time, existing_events or [], recovery_hours
        )
        
        final_score = pattern_score * recovery_penalty
        
        return min(1.0, max(0.1, final_score))  # Clamp between 0.1-1.0

    def generate_enhanced_slot_reasoning(self, start_time: datetime, 
                              duration: float, 
                              task_type: TaskType,
                              user_energy_pattern: List[float],
                              score: float) -> str:
        """Generate human-readable reasoning for slot selection using enhanced patterns"""
        hour = start_time.hour
        day_of_week = start_time.weekday()  # 0=Monday, 6=Sunday
        weekly_index = day_of_week * 24 + hour
        
        # Get pattern data from weekly arrays
        if weekly_index < len(task_type.weekly_habit_scores):
            preference = task_type.weekly_habit_scores[weekly_index]
        else:
            preference = 0.5  # Neutral fallback
            
        # Get confidence from 7x24 matrix
        if (day_of_week < len(task_type.slot_confidence) and 
            hour < len(task_type.slot_confidence[day_of_week])):
            confidence = task_type.slot_confidence[day_of_week][hour]
        else:
            confidence = 0.1  # Low confidence fallback
            
        cognitive_load = task_type.cognitive_load
        
        reasoning_parts = []
        
        # Time preference reasoning
        if preference > 0.7:
            reasoning_parts.append(f"High preference for {task_type.task_type} at {hour}:00")
        elif preference < 0.3:
            reasoning_parts.append(f"Low preference for {task_type.task_type} at {hour}:00")
        
        # Cognitive load reasoning with energy matching
        if cognitive_load > 0.7:
            reasoning_parts.append("High-cognitive task")
        elif cognitive_load < 0.3:
            reasoning_parts.append("Low-cognitive task")
        
        # Energy level reasoning
        energy_score = user_energy_pattern[weekly_index] if weekly_index < len(user_energy_pattern) else 0.5
        if energy_score > 0.7:
            reasoning_parts.append("High energy period")
        elif energy_score < 0.3:
            reasoning_parts.append("Low energy period")
        
        # Energy-cognitive fit reasoning
        if cognitive_load > 0.7:  # High cognitive load
            if energy_score > 0.7:
                reasoning_parts.append("High-cognitive task during high energy (excellent fit)")
            elif energy_score < 0.4:
                reasoning_parts.append("High-cognitive task during low energy (poor fit)")
        elif cognitive_load < 0.3:  # Low cognitive load
            if energy_score < 0.4:
                reasoning_parts.append("Low-cognitive task during low energy (good fit)")
            else:
                reasoning_parts.append("Low-cognitive task (flexible energy requirements)")
        
        # Confidence level
        if confidence > 0.8:
            reasoning_parts.append("High confidence in pattern")
        elif confidence < 0.3:
            reasoning_parts.append("Limited historical data")
        
        return " • ".join(reasoning_parts) if reasoning_parts else f"Score: {score:.2f}"

    def get_enhanced_pattern_insights(self, task_type: TaskType, user_energy_pattern: List[float]) -> Dict:
        """Extract key insights from enhanced behavioral patterns"""
        weekly_habit_scores = task_type.weekly_habit_scores
        
        # Find peak hours across the week (convert weekly index back to hours)
        peak_hours = []
        if weekly_habit_scores:
            for i, score in enumerate(weekly_habit_scores):
                if score > 0.7:
                    # Convert weekly index back to hour (0-23)
                    hour = i % 24
                    if hour not in peak_hours:
                        peak_hours.append(hour)
        
        # Find peak energy hours (highest energy scores)
        peak_energy_hours = []
        if user_energy_pattern:
            max_energy_score = max(user_energy_pattern)
            for i, energy_score in enumerate(user_energy_pattern):
                if energy_score > 0.7: # High energy threshold
                    if i not in peak_energy_hours:
                        peak_energy_hours.append(i)
        
        return {
            'best_hours': peak_hours,
            'cognitive_load': task_type.cognitive_load,
            'typical_duration': task_type.typical_duration,
            'recovery_needed': task_type.recovery_hours,
            'total_completions': task_type.completion_count,
            'peak_energy_hours': peak_energy_hours
        } 

    def generate_comprehensive_time_window_scores(self, user_id: str, 
                                                 user_energy_pattern: List[float],
                                                 search_window_days: int = 7) -> List[Dict]:
        """Generate scored time slots for entire window - foundation for unified scheduling"""
        
        time_slots = []
        start_time = self._get_next_clean_hour(datetime.now())
        end_time = start_time + timedelta(days=search_window_days)
        
        print(f"🕒 Generating time window scores from {start_time.strftime('%m/%d %H:%M')} to {end_time.strftime('%m/%d %H:%M')}")
        
        # Generate all possible 30-minute slots in window
        current_time = start_time
        slot_count = 0
        
        while current_time < end_time:
            slot = {
                'start_time': current_time,
                'end_time': current_time + timedelta(minutes=30),
                'is_free': True,
                'occupying_event': None,
                'occupying_event_priority': 0.0,
                'behavioral_scores': {},  # Will store scores for different task types
                'weekly_index': self._get_weekly_index(current_time)
            }
            time_slots.append(slot)
            current_time += timedelta(minutes=30)
            slot_count += 1
        
        print(f"✅ Generated {slot_count} time slots ({slot_count/48:.1f} days)")
        return time_slots
    
    def _get_weekly_index(self, dt: datetime) -> int:
        """Get weekly index (0-167) for a datetime"""
        # Convert Python weekday (0=Monday) to our schema (0=Sunday)
        python_weekday = dt.weekday()  # 0=Monday, 6=Sunday
        day_of_week = (python_weekday + 1) % 7  # Convert to 0=Sunday, 6=Saturday
        return day_of_week * 24 + dt.hour
    
    def mark_busy_slots(self, time_slots: List[Dict], existing_events: List[Dict]) -> None:
        """Mark time slots as busy and store occupying event information"""
        
        busy_count = 0
        for event in existing_events:
            event_start = event['scheduled_start']
            event_end = event['scheduled_end']
            event_priority = event.get('calculated_priority', 0.5)
            
            # Mark all overlapping slots as busy
            for slot in time_slots:
                if (slot['start_time'] < event_end and slot['end_time'] > event_start):
                    slot['is_free'] = False
                    slot['occupying_event'] = event
                    slot['occupying_event_priority'] = event_priority
                    busy_count += 1
        
        print(f"📅 Marked {busy_count} slots as busy from {len(existing_events)} existing events")
    
    def calculate_event_priority(self, request_or_event, task_type: TaskType = None) -> float:
        """Unified priority calculation using task_type.importance_score + deadline urgency"""
        
        # For existing events, use stored priority
        if isinstance(request_or_event, dict) and 'calculated_priority' in request_or_event:
            return request_or_event['calculated_priority']
        
        # For new requests, use task_type.importance_score if available
        if task_type:
            base_importance = task_type.importance_score  # Use learned importance from task type
        else:
            base_importance = getattr(request_or_event, 'importance_score', 0.5)
        
        # Calculate deadline urgency (if deadline provided)
        if hasattr(request_or_event, 'deadline') and request_or_event.deadline:
            time_to_deadline = (request_or_event.deadline - datetime.now()).total_seconds() / 3600
            if time_to_deadline <= 0:
                deadline_urgency = 1.0  # Past due
            elif time_to_deadline <= 24:
                deadline_urgency = 0.9  # Within 24 hours
            elif time_to_deadline <= 72:
                deadline_urgency = 0.7  # Within 3 days
            else:
                deadline_urgency = max(0.1, 1.0 - (time_to_deadline / (7*24)))  # Decay over week
        else:
            deadline_urgency = 0.3  # No deadline = moderate urgency
        
        # Combine: 70% learned importance, 30% deadline urgency
        priority = (base_importance * 0.7) + (deadline_urgency * 0.3)
        final_priority = min(1.0, max(0.1, priority))
        
        return final_priority
    
    def score_slots_for_task(self, time_slots: List[Dict], 
                           task_type: TaskType,
                           user_energy_pattern: List[float],
                           duration: float) -> None:
        """Score all time slots for a specific task type using our existing behavioral algorithm"""
        
        print(f"🎯 Scoring slots for task: {task_type.task_type} (cognitive load: {task_type.cognitive_load})")
        
        for slot in time_slots:
            # Use our existing enhanced scoring algorithm
            score = self._calculate_slot_behavioral_score(
                slot['start_time'], 
                duration,
                task_type, 
                user_energy_pattern
            )
            
            # Store the score for this task type
            slot['behavioral_scores'][task_type.task_type] = score
    
    def _calculate_slot_behavioral_score(self, start_time: datetime,
                                       duration: float,
                                       task_type: TaskType,
                                       user_energy_pattern: List[float]) -> float:
        """Calculate behavioral score for a single time slot (simplified version of our main algorithm)"""
        
        weekly_index = self._get_weekly_index(start_time)
        
        # Get habit preference
        if weekly_index < len(task_type.weekly_habit_scores):
            preference = task_type.weekly_habit_scores[weekly_index]
        else:
            preference = 0.5
        
        # Get energy score
        if weekly_index < len(user_energy_pattern):
            energy_score = user_energy_pattern[weekly_index]
        else:
            energy_score = 0.5
        
        # Apply energy-cognitive fit logic
        cognitive_load = task_type.cognitive_load
        if cognitive_load > 0.7:  # High cognitive load
            energy_cognitive_fit = energy_score  # Direct energy score
        elif cognitive_load < 0.3:  # Low cognitive load  
            energy_cognitive_fit = 1.0 - (1.0 - energy_score) * 0.5  # Boost low energy periods
        else:  # Medium cognitive load
            energy_cognitive_fit = energy_score * 0.8 + 0.2  # Slight boost
        
        # Combine factors (50/50 split)
        habit_component = preference * 0.5
        energy_component = energy_cognitive_fit * 0.5
        
        return habit_component + energy_component
    
    def group_slots_by_duration(self, time_slots: List[Dict], duration_hours: float) -> List[Dict]:
        """Group consecutive time slots to match requested duration"""
        
        duration_slots = int(duration_hours * 2)  # 30-minute slots
        slot_groups = []
        
        for i in range(len(time_slots) - duration_slots + 1):
            slots_group = time_slots[i:i + duration_slots]
            
            # Check if slots are consecutive
            consecutive = True
            for j in range(len(slots_group) - 1):
                if slots_group[j]['end_time'] != slots_group[j + 1]['start_time']:
                    consecutive = False
                    break
            
            if consecutive:
                slot_groups.append({
                    'start_time': slots_group[0]['start_time'],
                    'end_time': slots_group[-1]['end_time'],
                    'duration': duration_hours,
                    'slots': slots_group,
                    'total_score': 0.0,
                    'requires_rescheduling': False,
                    'reschedulable': True
                })
        
        return slot_groups
    
    def is_slot_group_viable(self, slot_group: Dict, new_request, task_type: TaskType) -> bool:
        """Check if a slot group is free or reschedulable by higher priority event"""
        
        new_event_priority = self.calculate_event_priority(new_request, task_type)
        rescheduling_required = False
        
        # Check each slot in the group
        for slot in slot_group['slots']:
            if not slot['is_free']:
                existing_priority = slot['occupying_event_priority']
                
                # Can we reschedule? Need buffer to prevent thrashing
                if new_event_priority <= existing_priority + 0.15:  # 0.15 buffer
                    slot_group['reschedulable'] = False
                    return False
                else:
                    rescheduling_required = True
        
        # Calculate total behavioral score for this slot group
        if task_type.task_type in slot_group['slots'][0]['behavioral_scores']:
            slot_group['total_score'] = sum(
                slot['behavioral_scores'][task_type.task_type] 
                for slot in slot_group['slots']
            ) / len(slot_group['slots'])  # Average score
        else:
            slot_group['total_score'] = 0.5  # Neutral fallback
        
        slot_group['requires_rescheduling'] = rescheduling_required
        return True 

    async def schedule_with_unified_scoring(self, user_id: str,
                                           user_energy_pattern: List[float],
                                           request: ScheduleEventRequest,
                                           existing_events: List[Dict],
                                           search_window_days: int = 7) -> Dict:
        """
        Unified scheduling algorithm using comprehensive time window scoring.
        Automatically pushes lower priority events when urgent events come in.
        """
        
        print(f"🚀 UNIFIED SCHEDULING: '{request.title}' (duration: {request.duration}h)")
        
        # 1. Find or create task type
        similar_task = await self.task_type_service.find_similar_task_type(
            user_id, f"{request.title} {request.description or ''}"
        )
        
        if similar_task and similar_task.similarity > 0.4:
            task_type = similar_task.task_type
            print(f"🎯 Using existing task type: '{task_type.task_type}' (similarity: {similar_task.similarity:.3f})")
        else:
            print(f"🆕 Creating new task type for: '{request.title}'")
            task_type = await self.task_type_service.create_task_type(
                user_id, request.title, request.description, self
            )
        
        # 2. Generate all time slots with scores
        all_slots = self.generate_comprehensive_time_window_scores(
            user_id, user_energy_pattern, search_window_days
        )
        
        # 3. Mark busy slots from existing events
        self.mark_busy_slots(all_slots, existing_events)
        
        # 4. Score all slots for this specific task
        self.score_slots_for_task(all_slots, task_type, user_energy_pattern, request.duration)
        
        # 5. Group slots by duration and find viable options
        slot_groups = self.group_slots_by_duration(all_slots, request.duration)
        viable_groups = []
        
        for group in slot_groups:
            if self.is_slot_group_viable(group, request, task_type):
                viable_groups.append(group)
        
        if not viable_groups:
            raise ValueError("No viable time slots found - all slots blocked by higher priority events")
        
        # 6. Sort by behavioral score and select best
        viable_groups.sort(key=lambda x: x['total_score'], reverse=True)
        best_group = viable_groups[0]
        
        print(f"✅ Found viable slot: {best_group['start_time'].strftime('%A %m/%d %H:%M')} "
              f"(score: {best_group['total_score']:.3f}, rescheduling: {best_group['requires_rescheduling']})")
        
        # 7. Execute scheduling (with rescheduling if needed)
        if best_group['requires_rescheduling']:
            return await self.execute_rescheduling_and_scheduling(
                best_group, request, task_type, user_id, user_energy_pattern, existing_events
            )
        else:
            return self.create_simple_scheduled_event(best_group, request, task_type, user_id)
    
    async def execute_rescheduling_and_scheduling(self, chosen_slot_group: Dict,
                                                new_request: ScheduleEventRequest,
                                                new_task_type: TaskType,
                                                user_id: str,
                                                user_energy_pattern: List[float],
                                                existing_events: List[Dict]) -> Dict:
        """Execute rescheduling of colliding events and schedule the new urgent event"""
        
        print(f"🔄 RESCHEDULING: Moving lower priority events to make room for urgent event")
        
        # 1. Identify events that need to be rescheduled
        events_to_reschedule = []
        for slot in chosen_slot_group['slots']:
            if not slot['is_free'] and slot['occupying_event'] not in events_to_reschedule:
                events_to_reschedule.append(slot['occupying_event'])
        
        print(f"📋 Need to reschedule {len(events_to_reschedule)} events:")
        for event in events_to_reschedule:
            print(f"   • {event['title']} (priority: {event.get('calculated_priority', 0.5):.3f})")
        
        # 2. Find alternative slots for each event to be rescheduled
        rescheduled_events = []
        updated_existing_events = [e for e in existing_events if e not in events_to_reschedule]
        
        for event_to_move in events_to_reschedule:
            print(f"🔍 Finding alternative slot for: {event_to_move['title']}")
            
            # Convert event back to a request-like object
            move_request = ScheduleEventRequest(
                title=event_to_move['title'],
                description=event_to_move.get('description', ''),
                duration=(event_to_move['scheduled_end'] - event_to_move['scheduled_start']).total_seconds() / 3600,
                preferred_date=event_to_move['scheduled_start']
            )
            
            # Find task type for the event being moved
            similar_task_for_move = await self.task_type_service.find_similar_task_type(
                user_id, f"{event_to_move['title']} {event_to_move.get('description', '')}"
            )
            
            if similar_task_for_move and similar_task_for_move.similarity > 0.4:
                move_task_type = similar_task_for_move.task_type
            else:
                # Fallback: create basic task type
                move_task_type = await self.task_type_service.create_task_type(
                    user_id, event_to_move['title'], event_to_move.get('description'), self
                )
            
            # Find alternative slot for this event (excluding the time we want to claim)
            alternative_slot = await self.find_alternative_slot_for_moved_event(
                move_request, move_task_type, user_energy_pattern, 
                updated_existing_events, chosen_slot_group
            )
            
            if alternative_slot:
                # Update the event with new timing
                event_to_move['scheduled_start'] = alternative_slot['start_time']
                event_to_move['scheduled_end'] = alternative_slot['end_time']
                event_to_move['calculated_priority'] = alternative_slot['total_score']
                
                rescheduled_events.append({
                    'original_event': event_to_move,
                    'new_start': alternative_slot['start_time'],
                    'new_end': alternative_slot['end_time'],
                    'score_change': alternative_slot['total_score'] - event_to_move.get('calculated_priority', 0.5)
                })
                
                # Add to updated events list for next iteration
                updated_existing_events.append(event_to_move)
                
                print(f"   ✅ Moved to: {alternative_slot['start_time'].strftime('%A %m/%d %H:%M')} "
                      f"(score: {alternative_slot['total_score']:.3f})")
            else:
                print(f"   ❌ Could not find alternative slot for: {event_to_move['title']}")
                # In real implementation, might need to expand search window or notify user
        
        # 3. Create the new urgent event and persist to database
        new_event_priority = self.calculate_event_priority(new_request, new_task_type)
        new_event_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "task_type_id": str(new_task_type.id),
            "title": new_request.title,
            "description": new_request.description,
            "scheduled_start": chosen_slot_group['start_time'],
            "scheduled_end": chosen_slot_group['end_time'],
            "calculated_priority": new_event_priority,
            "deadline": new_request.deadline
        }
        
        # Persist new event to database
        try:
            # Convert datetime objects to ISO format for database
            db_event_data = new_event_data.copy()
            db_event_data['scheduled_start'] = db_event_data['scheduled_start'].isoformat()
            db_event_data['scheduled_end'] = db_event_data['scheduled_end'].isoformat()
            if db_event_data['deadline']:
                db_event_data['deadline'] = db_event_data['deadline'].isoformat()
            
            db_result = self.task_type_service.supabase.table("events").insert(db_event_data).execute()
            print(f"✅ Created new urgent event in database: {new_event_data['id']}")
        except Exception as e:
            print(f"❌ Failed to create event in database: {e}")
            # Continue anyway - return the event data for testing
        
        # Update rescheduled events in database
        for rescheduled in rescheduled_events:
            original_event = rescheduled['original_event']
            try:
                update_result = self.task_type_service.supabase.table("events").update({
                    "scheduled_start": rescheduled['new_start'].isoformat(),
                    "scheduled_end": rescheduled['new_end'].isoformat(),
                    "calculated_priority": original_event['calculated_priority']
                }).eq("id", original_event['id']).execute()
                print(f"✅ Updated rescheduled event in database: {original_event['title']}")
            except Exception as e:
                print(f"❌ Failed to update rescheduled event in database: {e}")
                # Continue anyway
        
        return {
            "event": new_event_data,
            "scheduling_method": "unified_scoring_with_rescheduling",
            "slot_score": chosen_slot_group['total_score'],
            "rescheduled_events": rescheduled_events,
            "rescheduling_summary": {
                "events_moved": len(rescheduled_events),
                "new_event_priority": new_event_data['calculated_priority'],
                "average_moved_priority": sum(e['original_event'].get('calculated_priority', 0.5) for e in rescheduled_events) / len(rescheduled_events) if rescheduled_events else 0
            },
            "task_type_used": {
                "id": str(new_task_type.id),
                "name": new_task_type.task_type,
                "cognitive_load": new_task_type.cognitive_load
            }
        }
    
    async def find_alternative_slot_for_moved_event(self, move_request: ScheduleEventRequest,
                                                  move_task_type: TaskType,
                                                  user_energy_pattern: List[float],
                                                  current_events: List[Dict],
                                                  excluded_slot_group: Dict) -> Optional[Dict]:
        """Find the next best slot for an event that's being moved"""
        
        # Generate fresh time slots (broader search for alternatives)
        alternative_slots = self.generate_comprehensive_time_window_scores(
            "alternative_search", user_energy_pattern, search_window_days=14  # Broader search
        )
        
        # Mark busy slots from current events
        self.mark_busy_slots(alternative_slots, current_events)
        
        # Score slots for the task being moved
        self.score_slots_for_task(alternative_slots, move_task_type, user_energy_pattern, move_request.duration)
        
        # Group by duration
        alternative_groups = self.group_slots_by_duration(alternative_slots, move_request.duration)
        
        # Filter out the excluded time slot and find free slots only (no further rescheduling)
        viable_alternatives = []
        for group in alternative_groups:
            # Skip the slot we're trying to claim
            if (group['start_time'] >= excluded_slot_group['start_time'] and 
                group['start_time'] < excluded_slot_group['end_time']):
                continue
            
            # Only consider free slots (no cascading rescheduling)
            all_free = all(slot['is_free'] for slot in group['slots'])
            if all_free:
                group['total_score'] = sum(
                    slot['behavioral_scores'].get(move_task_type.task_type, 0.5) 
                    for slot in group['slots']
                ) / len(group['slots'])
                viable_alternatives.append(group)
        
        if viable_alternatives:
            # Return the best alternative
            viable_alternatives.sort(key=lambda x: x['total_score'], reverse=True)
            return viable_alternatives[0]
        else:
            return None
    
    def create_simple_scheduled_event(self, slot_group: Dict, 
                                    request: ScheduleEventRequest,
                                    task_type: TaskType,
                                    user_id: str) -> Dict:
        """Create event data for a simple scheduling (no rescheduling needed) and persist to database"""
        
        event_priority = self.calculate_event_priority(request, task_type)
        event_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "task_type_id": str(task_type.id),
            "title": request.title,
            "description": request.description,
            "scheduled_start": slot_group['start_time'],
            "scheduled_end": slot_group['end_time'],
            "calculated_priority": event_priority,
            "deadline": request.deadline
        }
        
        # Persist to database
        try:
            # Convert datetime objects to ISO format for database
            db_event_data = event_data.copy()
            db_event_data['scheduled_start'] = db_event_data['scheduled_start'].isoformat()
            db_event_data['scheduled_end'] = db_event_data['scheduled_end'].isoformat()
            if db_event_data['deadline']:
                db_event_data['deadline'] = db_event_data['deadline'].isoformat()
            
            db_result = self.task_type_service.supabase.table("events").insert(db_event_data).execute()
            print(f"✅ Created simple event in database: {event_data['id']}")
        except Exception as e:
            print(f"❌ Failed to create simple event in database: {e}")
            # Continue anyway - return the event data for testing
        
        return {
            "event": event_data,
            "scheduling_method": "unified_scoring_simple",
            "slot_score": slot_group['total_score'],
            "rescheduled_events": [],
            "task_type_used": {
                "id": str(task_type.id),
                "name": task_type.task_type,
                "cognitive_load": task_type.cognitive_load
            }
        } 