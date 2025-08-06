"""
Scheduler Service - Core Scheduling Algorithm
Uses behavioral patterns, energy levels, and deadlines to find optimal time slots
"""

import uuid
import math
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from models import TaskType, Event, ScheduleEventRequest, SchedulingResult
from task_type_service import TaskTypeService
from db_service import DatabaseService

# Configure logging for scheduler service
logger = logging.getLogger(__name__)
class SchedulerService:
    def __init__(self, task_type_service: TaskTypeService):
        self.task_type_service = task_type_service
        self.db_service = DatabaseService(task_type_service.supabase)
    
    # ============================================================================
    # PUBLIC SCHEDULING METHODS
    # ============================================================================
    
    # @tool  # Removed - tool wrapper is created in API layer
    async def schedule_with_pattern(self,
                                   user_id: str,
                                   start: str = None,
                                   end: str = None,
                                   timeZone: str = None,
                                   summary: str = None,
                                   description: str = None,
                                   location: str = None,
                                   category: str = None,
                                   duration: float = 1.0,
                                   importance_score: float = 0.5,
                                   available_periods: Optional[List[Tuple[datetime, datetime]]] = None,
                                   openai_client = None,
                                   memory_service = None) -> Dict:
        """
        Creates a NEW calendar event with the provided details using pattern-based scheduling.
        Routes to LLM if similarity < 0.4 threshold.

        Args:
            user_id (str): The user's ID to use their specific credentials
            start (str): Event start time in ISO 8601 format.
            end (str): Event end time in ISO 8601 format.
            timeZone (str, optional): User timezone as IANA Time Zone name.
            summary (str, optional): Short title/description of the event. Defaults to None.
            location (str, optional): Location of the event. Defaults to None.
            category (str): If user provide a start time or a fixed time, the category will be "Event", 
                          else if no time or only a deadline, the category will be "Task".
            duration: Duration in hours (default: 1.0)
            importance_score: Task importance 0.0-1.0 (default: 0.5)
            available_periods: Optional time periods to search within
            openai_client: OpenAI client for LLM fallback
            memory_service: Memory service for preference storage
            
        Returns:
            str: Brief message if successful with Firebase ID 
        """
        
        logger.info(f"📅 SCHEDULE START: User {user_id}, task '{summary}', duration {duration}h")
        logger.info(f"📅 SCHEDULE PARAMS: start={start}, end={end}, timezone={timeZone}")
        logger.info(f"📅 SCHEDULE PARAMS: importance={importance_score}, description='{description or 'None'}'")
        
        # Validate required parameters
        if not summary:
            logger.error("📅 SCHEDULE ERROR: Summary/title is required")
            raise ValueError("Summary/title is required for scheduling")
        
        # Determine scheduling approach and get optimal slot
        if start and end:
            # Direct scheduling - user provided specific times
            logger.info(f"🎯 DIRECT SCHEDULING: User provided start/end times")
            
            try:
                from datetime import datetime
                scheduled_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
                scheduled_end = datetime.fromisoformat(end.replace('Z', '+00:00'))
                logger.info(f"🎯 DIRECT SCHEDULING: Parsed times - start: {scheduled_start}, end: {scheduled_end}")
                
                # Use provided times as optimal slot
                optimal_slot = {
                    'start_time': scheduled_start,
                    'end_time': scheduled_end
                }
                task_type_id = None
                scheduling_method = "direct"
                logger.info(f"🎯 DIRECT SCHEDULING: Using method '{scheduling_method}'")
                
            except Exception as e:
                logger.error(f"🎯 DIRECT SCHEDULING ERROR: Invalid time format: {e}")
                raise ValueError(f"Invalid start/end time format: {e}")
        else:
            # Pattern-based scheduling - find optimal time slot
            logger.info(f"🎯 PATTERN-BASED SCHEDULING: Finding optimal time slot")
            
            # Fetch existing events from database within available periods
            time_periods = self._setup_time_periods(available_periods)
            logger.info(f"📅 TIME PERIODS: Setup {len(time_periods)} periods")
            
            existing_events = await self._fetch_existing_events(user_id, time_periods)
            logger.info(f"📅 EXISTING EVENTS: Found {len(existing_events)} events")
            
            # Find similar task type via RAG/similarity search
            logger.info(f"🔍 SIMILARITY SEARCH: Looking for similar task types")
            similar_task = await self.task_type_service.find_similar_task_type(
                user_id, summary
            )
            
            if similar_task and similar_task.similarity >= 0.4:
                task_type = similar_task.task_type
                logger.info(f"🎯 PATTERN SCHEDULING: Using '{task_type.task_type}' (similarity: {similar_task.similarity:.3f}, completions: {task_type.completion_count})")
                
                # Find optimal slot using behavioral patterns + energy + cognitive load
                logger.info(f"🔍 SLOT FINDING: Searching for optimal slot")
                optimal_result = await self.find_optimal_slot(
                    user_id, task_type, duration, time_periods, existing_events
                )
                
                if not optimal_result:
                    logger.error("🔍 SLOT FINDING ERROR: No available time slots found")
                    raise ValueError("No available time slots found using pattern-based scheduling")
                
                optimal_slot = optimal_result['optimal_slot']
                task_type_id = str(task_type.id)
                scheduling_method = "pattern_based"
                logger.info(f"🎯 PATTERN SCHEDULING: Found slot at {optimal_slot['start_time']} - {optimal_slot['end_time']}")
                
            else:
                # Auto-fallback to LLM scheduling when similarity < 0.4
                similarity_msg = f"similarity: {similar_task.similarity:.3f}" if similar_task else "no matches found"
                logger.info(f"🔄 ROUTING TO LLM: {similarity_msg} < 0.4 threshold")
                
                user_preferences = f"Creating task: '{summary}'. " + \
                                 "Please analyze optimal scheduling based on patterns, energy, and cognitive load."
                
                # Create a request object for LLM compatibility
                from models import ScheduleEventRequest
                request = ScheduleEventRequest(
                    title=summary,
                    description=description,
                    duration=duration,
                    importance_score=importance_score
                )
                
                return await self.schedule_with_llm(
                    user_id=user_id,
                    request=request,
                    user_preferences=user_preferences,
                    existing_events=existing_events,
                    available_periods=available_periods,
                    openai_client=openai_client,
                    memory_service=memory_service
                )
        
        # Single event creation for both direct and pattern-based scheduling
        logger.info(f"💾 DATABASE: Creating event in database")
        try:
            event_id = await self.db_service.create_event(
                user_id=user_id,
                title=summary,
                description=description,
                scheduled_start=optimal_slot['start_time'],
                scheduled_end=optimal_slot['end_time'],
                task_type_id=task_type_id,
                calculated_priority=importance_score
            )
            logger.info(f"💾 DATABASE: Event created successfully with ID: {event_id}")
        except Exception as e:
            logger.error(f"💾 DATABASE ERROR: Failed to create event: {e}")
            raise
        
        logger.info(f"📅 SCHEDULE SUCCESS: Event {event_id} scheduled using {scheduling_method}")
        return event_id

    async def schedule_with_llm(self,
                               user_id: str,
                               request: ScheduleEventRequest,
                               user_preferences: str,
                               existing_events: List[Dict],
                               available_periods: Optional[List[Tuple[datetime, datetime]]] = None,
                               openai_client = None,
                               memory_service = None) -> Dict:
        """
        LLM-based scheduling with behavioral context for semantic understanding
        """
        
        # 1. Find/create task type for behavioral context
        task_type = await self._find_or_create_task_type(user_id, request)
        
        # 2. Set up time periods with deadline awareness
        time_periods = self._setup_time_periods(available_periods)
        
        # 3. Extract free slots from calendar data
        free_slots = self._extract_free_slots_from_periods(
            time_periods, existing_events, request.duration
        )
        
        # 4. Prepare comprehensive context for LLM
        llm_context = await self._prepare_llm_context(
            user_id, task_type, user_preferences, free_slots, existing_events, request, memory_service
        )
        
        # 5. Get LLM scheduling decision
        if not openai_client:
            raise ValueError("OpenAI client required for LLM scheduling")
        
        llm_response = await self._call_llm_for_scheduling(openai_client, llm_context, free_slots)
        
        # 6. Create event from LLM decision
        selected_slot = self._parse_llm_scheduling_response(llm_response, free_slots)
        
        # 7. Create event in database (was missing!)
        logger.info(f"💾 DATABASE: Creating LLM-scheduled event in database")
        try:
            event_id = await self.db_service.create_event(
                user_id=user_id,
                title=request.title,
                description=request.description,
                scheduled_start=selected_slot['start_time'],
                scheduled_end=selected_slot['end_time'],
                task_type_id=str(task_type.id),
                calculated_priority=selected_slot.get('priority', 0.8)
            )
            logger.info(f"💾 DATABASE: LLM event created successfully with ID: {event_id}")
        except Exception as e:
            logger.error(f"💾 DATABASE ERROR: Failed to create LLM event: {e}")
            raise
        
        # 8. Store new preferences if memory service available
        if memory_service and user_preferences:
            await self._store_scheduling_preferences(
                user_id, user_preferences, task_type, llm_response, memory_service, openai_client
            )
        
        return event_id

    async def schedule_event(self, user_id: str, 
                           request: ScheduleEventRequest,
                           available_periods: Optional[List[Tuple[datetime, datetime]]] = None) -> Dict:
        """
        Legacy scheduling method - calls schedule_with_pattern for backward compatibility
        """
        return await self.schedule_with_pattern(
            user_id=user_id,
            request=request,
            existing_events=[],
            available_periods=available_periods
        )
    
    # ============================================================================
    # CORE SCORING AND PATTERN ANALYSIS
    # ============================================================================
    
    async def calculate_slot_score(self, start_time: datetime, 
                           duration: float, 
                           task_type: TaskType,
                           user_id: str,
                           existing_events: List[Dict] = None) -> float:
        """Calculate comprehensive score: 50% habit patterns + 50% energy matching, with confidence multiplier"""
        
        weekly_habit_scores = task_type.weekly_habit_scores  # 168-hour task-specific patterns
        slot_confidence = task_type.slot_confidence          # 7x24 confidence matrix
        cognitive_load = task_type.cognitive_load
        
        # Get user's energy pattern from database
        user_energy_pattern = await self._get_user_energy_pattern(user_id)
        
        total_score = 0
        total_weight = 0
        
        # Score each hour the task spans
        current_hour = start_time.hour
        remaining_duration = duration
        
        while remaining_duration > 0:
            hour_fraction = min(1.0, remaining_duration)
            hour_index = current_hour % 24
            
            # Convert to our day schema (0=Sunday)
            python_weekday = start_time.weekday()  # 0=Monday, 6=Sunday
            day_of_week = (python_weekday + 1) % 7  # Convert to 0=Sunday, 6=Saturday
            weekly_index = day_of_week * 24 + hour_index
            
            # 1. Get habit pattern score (50%)
            if weekly_index < len(weekly_habit_scores):
                habit_score = weekly_habit_scores[weekly_index]
            else:
                habit_score = 0.5  # Neutral fallback
            
            # 2. Get user's energy level at this time (50%)
            if weekly_index < len(user_energy_pattern):
                energy_level = user_energy_pattern[weekly_index]
            else:
                energy_level = 0.5  # Neutral fallback
            
            # 3. Calculate cognitive-energy matching
            if cognitive_load > 0.7 and energy_level < 0.6:
                # High cognitive task during low energy - penalty
                energy_match_score = energy_level * 0.6  # Penalty
            elif cognitive_load < 0.3 and energy_level > 0.8:
                # Low cognitive task during high energy - waste penalty
                energy_match_score = energy_level * 0.75  # Slight penalty
            else:
                # Good match or neutral
                energy_match_score = energy_level
            
            # 4. Combine 50-50: habit patterns + energy matching
            combined_score = (habit_score * 0.5) + (energy_match_score * 0.5)
            
            # 5. Get confidence multiplier
            if (day_of_week < len(slot_confidence) and 
                hour_index < len(slot_confidence[day_of_week])):
                confidence = slot_confidence[day_of_week][hour_index]
            else:
                confidence = 0.1  # Low confidence fallback
            
            # 6. Apply confidence as multiplier
            final_hour_score = combined_score * max(0.1, confidence) * hour_fraction
            
            total_score += final_hour_score
            total_weight += confidence * hour_fraction
            remaining_duration -= hour_fraction
            current_hour += 1
        
        # Calculate final score
        final_score = total_score / total_weight if total_weight > 0 else 0.5
        return min(1.0, max(0.1, final_score))  # Clamp between 0.1-1.0

    async def _get_user_energy_pattern(self, user_id: str) -> List[float]:
        """Get user's 168-hour energy pattern from database"""
        try:
            result = self.task_type_service.supabase.table("users") \
                .select("weekly_energy_pattern") \
                .eq("id", user_id) \
                .execute()
            
            if result.data and result.data[0].get("weekly_energy_pattern"):
                return result.data[0]["weekly_energy_pattern"]
            else:
                # Return neutral energy pattern (168 hours of 0.5)
                return [0.5] * 168
        except Exception as e:
            logger.warning(f"Could not get user energy pattern: {e}, using neutral")
            return [0.5] * 168



    def get_pattern_insights(self, task_type: TaskType) -> Dict:
        """Extract key insights from behavioral patterns"""
        weekly_habit_scores = task_type.weekly_habit_scores
        
        # Find peak hours across the week
        peak_hours = []
        if weekly_habit_scores:
            for i, score in enumerate(weekly_habit_scores):
                if score > 0.7:
                    hour = i % 24
                    if hour not in peak_hours:
                        peak_hours.append(hour)
        
        return {
            'best_hours': peak_hours,
            'peak_energy_hours': peak_hours,  # Same as best_hours since patterns include energy
            'cognitive_load': task_type.cognitive_load,
            'typical_duration': task_type.typical_duration,
            'recovery_needed': task_type.recovery_hours,
            'total_completions': task_type.completion_count
        }
    
    def get_cognitive_energy_analysis(self, cognitive_load: float, energy_level: float) -> Dict:
        """Analyze cognitive-energy matching and return penalty info"""
        analysis = {
            'cognitive_load': cognitive_load,
            'energy_level': energy_level,
            'penalty_factor': 1.0,
            'penalty_reason': None,
            'match_quality': 'optimal'
        }
        
        if cognitive_load > 0.7 and energy_level < 0.6:
            # High cognitive load during low energy - major penalty
            analysis['penalty_factor'] = 0.6
            analysis['penalty_reason'] = "High-cognitive task during low energy period"
            analysis['match_quality'] = 'poor'
        elif cognitive_load < 0.3 and energy_level > 0.8:
            # Low cognitive load during high energy - waste penalty
            analysis['penalty_factor'] = 0.75
            analysis['penalty_reason'] = "Low-cognitive task wastes peak energy period"
            analysis['match_quality'] = 'wasteful'
        elif cognitive_load > 0.6 and energy_level > 0.7:
            # High cognitive + high energy = perfect match
            analysis['match_quality'] = 'perfect'
        elif cognitive_load < 0.4 and energy_level < 0.5:
            # Low cognitive + low energy = good match
            analysis['match_quality'] = 'good'
        
        return analysis
    
    # ============================================================================
    # SLOT FINDING AND AVAILABILITY
    # ============================================================================

    async def find_optimal_slot(self, user_id: str, 
                              task_type: TaskType,
                              duration: float,
                              available_periods: List[Tuple[datetime, datetime]],
                              existing_events: List[Dict]) -> Optional[Dict]:
        """Find optimal time slot using comprehensive pattern analysis"""
        
        logger.info(f"🔍 OPTIMAL SLOT: Searching for {duration}h slot for '{task_type.task_type}'")
        logger.info(f"🔍 OPTIMAL SLOT: {len(available_periods)} periods, {len(existing_events)} existing events")
        
        slot_candidates = []
        
        # Search within each available time period
        for period_start, period_end in available_periods:
            current_time = period_start.replace(minute=0, second=0, microsecond=0)
            
            # Ensure we start at the beginning of the period or later
            if current_time < period_start:
                current_time = period_start
            
            while current_time + timedelta(hours=duration) <= period_end:
                # Check basic availability
                if self.is_slot_available(current_time, duration, existing_events):
                    
                    # Calculate comprehensive score
                    score = await self.calculate_slot_score(
                        current_time, duration, task_type, user_id, existing_events
                    )
                    
                    # Generate detailed reasoning
                    reasoning = self.generate_slot_reasoning(
                        current_time, duration, task_type, score
                    )
                    
                    slot_candidates.append({
                        'start_time': current_time,
                        'end_time': current_time + timedelta(hours=duration),
                        'score': score,
                        'reasoning': reasoning,
                        'period': f"{period_start.strftime('%m/%d %H:%M')}-{period_end.strftime('%H:%M')}"
                    })
                
                # Move to next 30-minute interval
                current_time += timedelta(minutes=30)
        
        if not slot_candidates:
            logger.warning(f"🔍 OPTIMAL SLOT: No candidates found")
            return None
        
        # Sort by score and return best options
        best_slots = sorted(slot_candidates, key=lambda x: x['score'], reverse=True)
        logger.info(f"🔍 OPTIMAL SLOT: Found {len(slot_candidates)} candidates, best score: {best_slots[0]['score']:.3f}")
        logger.info(f"🔍 OPTIMAL SLOT: Best slot: {best_slots[0]['start_time']} - {best_slots[0]['end_time']}")
        
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
        day_of_week = start_time.weekday()
        weekly_index = day_of_week * 24 + hour
        
        reasoning_parts = []
        
        # Get pattern data
        if weekly_index < len(task_type.weekly_habit_scores):
            preference = task_type.weekly_habit_scores[weekly_index]
        else:
            preference = 0.5
            
        cognitive_load = task_type.cognitive_load
        
        # Pattern reasoning
        if preference > 0.7:
            reasoning_parts.append(f"High preference for {task_type.task_type} at {hour}:00")
        elif preference < 0.3:
            reasoning_parts.append(f"Low preference at {hour}:00")
        
        # Cognitive load reasoning
        if cognitive_load > 0.7:
            reasoning_parts.append("High-cognitive task")
            if preference > 0.6:
                reasoning_parts.append("Good energy match")
            else:
                reasoning_parts.append("Low energy penalty (-40%)")
        elif cognitive_load < 0.3:
            reasoning_parts.append("Low-cognitive task")
            if preference > 0.8:
                reasoning_parts.append("Peak energy waste penalty (-25%)")
            else:
                reasoning_parts.append("Good for flexible scheduling")
        
        return " • ".join(reasoning_parts) if reasoning_parts else f"Score: {score:.2f}"

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
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _get_next_clean_hour(self, dt: datetime) -> datetime:
        """Get next clean hour boundary from given datetime"""
        if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0:
            return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            return dt.replace(minute=0, second=0, microsecond=0)

    def _setup_time_periods(self, available_periods: Optional[List[Tuple[datetime, datetime]]] = None) -> List[Tuple[datetime, datetime]]:
        """Set up time periods for scheduling"""
        if available_periods is None:
            # Default: search next 7 days starting from next clean hour
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
            print(f"📅 Searching in {len(available_periods)} periods: {periods_str}")
        
        return available_periods

    async def _fetch_existing_events(self, user_id: str, time_periods: List[Tuple[datetime, datetime]]) -> List[Dict]:
        """Fetch existing events from database within the specified time periods"""
        try:
            # Get the overall time range to query
            if not time_periods:
                return []
            
            earliest_start = min(period[0] for period in time_periods)
            latest_end = max(period[1] for period in time_periods)
            
            # Query events from database
            result = self.task_type_service.supabase.table("events") \
                .select("*") \
                .eq("user_id", user_id) \
                .gte("scheduled_start", earliest_start.isoformat()) \
                .lte("scheduled_end", latest_end.isoformat()) \
                .execute()
            
            existing_events = []
            for event_data in result.data:
                # Convert to our expected format
                existing_events.append({
                    "id": event_data["id"],
                    "user_id": event_data["user_id"],
                    "task_type_id": event_data.get("task_type_id"),
                    "title": event_data["title"],
                    "description": event_data.get("description"),
                    "scheduled_start": datetime.fromisoformat(event_data["scheduled_start"]),
                    "scheduled_end": datetime.fromisoformat(event_data["scheduled_end"]),
                    "cognitive_load": event_data.get("cognitive_load", 0.5)
                })
            
            print(f"📅 Found {len(existing_events)} existing events in time range")
            return existing_events
            
        except Exception as e:
            print(f"⚠️ Could not fetch existing events: {e}")
            return []  # Return empty list on error

    async def _find_or_create_task_type(self, user_id: str, request: ScheduleEventRequest) -> TaskType:
        """Find similar task type or create new one"""
        similar_task = await self.task_type_service.find_similar_task_type(
            user_id, request.title
        )
        
        if similar_task and similar_task.similarity > 0.4:
            print(f"🎯 LLM SCHEDULING: Using existing task type '{similar_task.task_type.task_type}' (completions: {similar_task.task_type.completion_count})")
            return similar_task.task_type
        else:
            print(f"🆕 Creating new task type for LLM: '{request.title}'")
            return await self.task_type_service.create_task_type(
                user_id, request.title, request.description
            )

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
    
    async def _prepare_llm_context(self,
                                  user_id: str,
                                  task_type: TaskType,
                                  user_preferences: str,
                                  free_slots: List[Dict],
                                  existing_events: List[Dict],
                                  request: ScheduleEventRequest,
                                  memory_service = None) -> str:
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
"""
        if memory_service:
            from mem0_service import Mem0Service
            mem0_service = Mem0Service(memory_service, self.task_type_service)
            mem0_context = await mem0_service.query_scheduling_context(
                user_id, task_type, request
            )
            context += mem0_context if mem0_context else "No previous patterns found"
        else:
            context += "Memory service not available to query historical patterns."

        context += f"""

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
    
    async def _store_scheduling_preferences(self, user_id: str, user_preferences: str, 
                                           task_type: TaskType, llm_response: Dict, 
                                           memory_service, openai_client):
        """Store new scheduling preferences from user input"""
        try:
            from mem0_service import Mem0Service
            mem0_service = Mem0Service(memory_service, self.task_type_service)
            await mem0_service.store_scheduling_preferences(
                user_id, user_preferences, task_type, llm_response.get('detected_patterns', []), openai_client
            )
        except Exception as e:
            print(f"⚠️ Could not store scheduling preferences: {e}")

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