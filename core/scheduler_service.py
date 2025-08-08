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
from langchain_core.tools import tool
from models import TaskType, Event, SchedulingResult
from .task_type_service import TaskTypeService
from .db_service import DatabaseService
from .config import get_openai_client

# Configure logging for scheduler service
logger = logging.getLogger(__name__)
class SchedulerService:
    def __init__(self, task_type_service: TaskTypeService):
        self.task_type_service = task_type_service
        self.db_service = DatabaseService(task_type_service.supabase)
    
    # ============================================================================
    # PUBLIC SCHEDULING METHODS
    # ============================================================================
    
    @tool
    async def schedule_with_pattern(self,
                                   user_id: str,
                                   start: str | None = None,
                                   end: str | None = None,
                                   timeZone: str | None = None,
                                   summary: str | None = None,
                                   description: str | None = None,
                                   location: str | None = None,
                                   category: str | None = None,
                                   duration: float = 1.0,
                                   importance_score: float = 0.5,
                                   deadline: str | None = None,
                                   available_periods: str | None = None) -> str:
        """
        Creates a NEW calendar event with the provided details using pattern-based scheduling.
        Routes to LLM if similarity < 0.4 threshold.

        Args:
            user_id (str): The user's ID to use their specific credentials
            start (str, optional): Event start time in ISO 8601 format. Defaults to None.
            end (str, optional): Event end time in ISO 8601 format. Defaults to None.
            timeZone (str, optional): User timezone as IANA Time Zone name. Defaults to None.
            summary (str, optional): Short title/description of the event. Defaults to None.
            description (str, optional): Detailed description of the event. Defaults to None.
            location (str, optional): Location of the event. Defaults to None.
            category (str, optional): If user provide a start time or a fixed time, the category will be "Event", 
                          else if no time or only a deadline, the category will be "Task". Defaults to None.
            duration (float): Duration in hours. Defaults to 1.0.
            importance_score (float): Task importance 0.0-1.0 (0.0=low priority, 1.0=critical). Defaults to 0.5.
            deadline (str, optional): Optional deadline in ISO 8601 format. Defaults to None.
            available_periods (str, optional): Time periods to search within. Format: "start1,end1;start2,end2" 
                          where each date is in ISO 8601 format. Example: "2024-01-15T09:00:00,2024-01-15T17:00:00;2024-01-16T09:00:00,2024-01-16T17:00:00". Defaults to None.
            
        Returns:
            str: Event ID if successful, or error message if failed
        """
        
        logger.info("🔧 SCHEDULER TOOL INVOKED")
        logger.info(f"📅 SCHEDULE START: User {user_id}, task '{summary}', duration {duration}h")
        logger.info(f"📅 SCHEDULE PARAMS: start={start}, end={end}, timezone={timeZone}")
        logger.info(f"📅 SCHEDULE PARAMS: importance={importance_score}, deadline={deadline}")
        logger.info(f"📅 SCHEDULE PARAMS: location='{location}', category='{category}'")
        logger.info(f"📅 SCHEDULE PARAMS: available_periods='{available_periods}'")
        logger.info(f"📅 SCHEDULE PARAMS: description='{description or 'None'}'")
        
        # Parse deadline if provided
        parsed_deadline = None
        if deadline:
            try:
                parsed_deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
                logger.info(f"📅 DEADLINE: Parsed deadline as {parsed_deadline}")
            except Exception as e:
                logger.warning(f"📅 DEADLINE WARNING: Invalid deadline format '{deadline}': {e}")
        
        # Validate required parameters
        if not summary:
            logger.error("📅 SCHEDULE ERROR: Summary/title is required")
            raise ValueError("Summary/title is required for scheduling")
        
        # Determine if this is auto-schedule or direct schedule
        is_auto_schedule = not (start and end)
        
        # Setup common variables
        task_type_id = None
        optimal_result = {}
        
        # Fetch existing events for collision detection (for all scheduling types)
        if is_auto_schedule:
            # Parse available_periods string if provided
            parsed_periods = None
            if available_periods:
                parsed_periods = self._parse_available_periods_string(available_periods)
            time_periods = self._setup_time_periods(parsed_periods)
            all_existing_events = await self._fetch_existing_events(user_id, time_periods)
        else:
            # For direct scheduling, fetch events in a broader range for collision detection
            scheduled_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
            scheduled_end = datetime.fromisoformat(end.replace('Z', '+00:00'))
            query_start = scheduled_start - timedelta(hours=12)
            query_end = scheduled_end + timedelta(hours=12)
            time_periods = [(query_start, query_end)]
            all_existing_events = await self._fetch_existing_events(user_id, time_periods)
        
        logger.info(f"📅 EXISTING EVENTS: Found {len(all_existing_events)} events for collision check")
        
        if is_auto_schedule:
            # Auto-scheduling: Pattern-based scheduling
            logger.info(f"🎯 AUTO-SCHEDULING: Finding optimal time slot using patterns")
            
            existing_events = all_existing_events  # Use all events for pattern scheduling
            
            # Find similar task type via RAG/similarity search
            logger.info(f"🔍 SIMILARITY SEARCH: Looking for similar task types")
            similar_task = await self.task_type_service.find_similar_task_type(user_id, summary)
            
            if similar_task and similar_task.similarity >= 0.4:
                task_type = similar_task.task_type
                logger.info(f"🎯 PATTERN SCHEDULING: Using '{task_type.task_type}' (similarity: {similar_task.similarity:.3f}, completions: {task_type.completion_count})")
                task_type_id = str(task_type.id)
            else:
                # Create new task type when similarity < 0.4
                similarity_msg = f"similarity: {similar_task.similarity:.3f}" if similar_task else "no matches found"
                logger.info(f"🆕 CREATING NEW TASK TYPE: {similarity_msg} < 0.4 threshold")
                task_type = await self.task_type_service.create_task_type(user_id, summary, description)
                logger.info(f"🆕 NEW TASK TYPE: Created '{task_type.task_type}' with LLM-generated patterns")
                task_type_id = str(task_type.id)
            
            # Find optimal slot using behavioral patterns
            optimal_result = await self.find_optimal_slot(
                user_id, task_type, duration, time_periods, existing_events, importance_score, parsed_deadline
            )
            
            if not optimal_result:
                logger.error("🔍 SLOT FINDING ERROR: No available time slots found")
                raise ValueError("No available time slots found using pattern-based scheduling")
                
            optimal_slot = optimal_result['optimal_slot']
            scheduling_method = "auto_schedule"
            
        else:
            # Direct scheduling - user provided specific times with collision detection
            logger.info(f"🎯 DIRECT SCHEDULING: User provided start/end times")
            
            try:
                scheduled_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
                scheduled_end = datetime.fromisoformat(end.replace('Z', '+00:00'))
                logger.info(f"🎯 DIRECT SCHEDULING: Parsed times - start: {scheduled_start}, end: {scheduled_end}")
                
                # For direct scheduling, always displace conflicting movable events (100% priority)
                conflicting_events = self._find_conflicting_events(scheduled_start, scheduled_end, all_existing_events)
                movable_conflicts = [e for e in conflicting_events if e.get('task_type_id')]  # Only movable events
                
                if movable_conflicts:
                    logger.info(f"🔄 DIRECT SCHEDULE DISPLACEMENT: Moving {len(movable_conflicts)} conflicting auto-scheduled events")
                    await self._displace_conflicting_events(movable_conflicts)
                
                optimal_slot = {
                    'start_time': scheduled_start,
                    'end_time': scheduled_end,
                    'fit_score': 1.0  # Direct schedule gets perfect fit score
                }
                scheduling_method = "direct_schedule"
                
            except Exception as e:
                logger.error(f"🎯 DIRECT SCHEDULING ERROR: Invalid time format: {e}")
                raise ValueError(f"Invalid start/end time format: {e}")
        
        # Single event creation for both auto and direct scheduling
        logger.info(f"DATABASE_CREATE: Creating event in database")
        try:
            # Prepare alternative slots for storage
            alternative_slots = []
            if 'alternatives' in optimal_result:
                for alt in optimal_result['alternatives'][:5]:  # Store top 5 alternatives
                    alternative_slots.append({
                        "start": alt['start_time'].isoformat(),
                        "end": alt['end_time'].isoformat(),
                        "score": alt.get('full_score', alt.get('fit_score', 0.5)),
                        "fit_score": alt.get('fit_score', 0.5),
                        "priority_score": alt.get('priority_score', importance_score)
                    })
            
            event_id = await self.db_service.create_event(
                user_id=user_id,
                title=summary,
                description=description,
                scheduled_start=optimal_slot['start_time'],
                scheduled_end=optimal_slot['end_time'],
                task_type_id=task_type_id,
                calculated_priority=importance_score,
                deadline=parsed_deadline
            )
            logger.info(f"DATABASE_CREATE_SUCCESS: Event created with ID: {event_id}, {len(alternative_slots)} alternatives stored")
        except Exception as e:
            logger.error(f"DATABASE_CREATE_ERROR: Failed to create event: {e}")
            raise
        
        logger.info(f"📅 SCHEDULE SUCCESS: Event {event_id} scheduled using {scheduling_method}")
        return event_id



    
    # ============================================================================
    # CORE SCORING AND PATTERN ANALYSIS
    # ============================================================================
    
    async def calculate_fit_score(self, start_time: datetime, 
                           duration: float, 
                           task_type: TaskType,
                           user_id: str,
                           user_energy_pattern: Optional[List[float]] = None
                        ) -> float:
        """Calculate comprehensive score: 50% habit patterns + 50% energy matching, with confidence multiplier"""
        
        weekly_habit_scores = task_type.weekly_habit_scores  # 168-hour task-specific patterns
        slot_confidence = task_type.slot_confidence          # 7x24 confidence matrix
        cognitive_load = task_type.cognitive_load
        
        # Get user's energy pattern from database or use provided pattern
        if user_energy_pattern is None:
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

    def filter_available_periods(self, available_periods: List[Tuple[datetime, datetime]], 
                               fixed_events: List[Dict]) -> List[Tuple[datetime, datetime]]:
        """Filter out time periods blocked by fixed (immovable) events"""
        filtered_periods = []
        
        for period_start, period_end in available_periods:
            current_segments = [(period_start, period_end)]
            
            # Remove segments blocked by fixed events
            for event in fixed_events:
                if not event.get('task_type_id'):  # Fixed event (no task_type_id)
                    event_start = event['scheduled_start']
                    event_end = event['scheduled_end']
                    
                    new_segments = []
                    for seg_start, seg_end in current_segments:
                        # Check if event overlaps with this segment
                        if event_start < seg_end and event_end > seg_start:
                            # Add segment before event
                            if seg_start < event_start:
                                new_segments.append((seg_start, event_start))
                            # Add segment after event
                            if event_end < seg_end:
                                new_segments.append((event_end, seg_end))
                        else:
                            # No overlap, keep segment
                            new_segments.append((seg_start, seg_end))
                    current_segments = new_segments
            
            # Add all valid segments (minimum 30 minutes)
            for seg_start, seg_end in current_segments:
                if (seg_end - seg_start).total_seconds() >= 1800:  # 30 minutes
                    filtered_periods.append((seg_start, seg_end))
        
        return filtered_periods

    async def reschedule_using_alternatives(self, event: Dict, existing_events: List[Dict]) -> Optional[Dict]:
        """Try to reschedule event using stored alternative slots"""
        alternative_slots = event.get('alternative_slots', [])
        
        if not alternative_slots:
            logger.info(f"RESCHEDULE_NO_ALTERNATIVES: Event '{event['title']}' has no stored alternatives")
            return None
        
        logger.info(f"RESCHEDULE_CHECK_ALTERNATIVES: Checking {len(alternative_slots)} stored alternatives for '{event['title']}'")
        
        # Check each alternative slot for conflicts
        for i, alt_slot in enumerate(alternative_slots):
            try:
                alt_start = datetime.fromisoformat(alt_slot['start'])
                alt_end = datetime.fromisoformat(alt_slot['end'])
                
                # Check if this alternative conflicts with any existing events
                has_conflict = False
                for existing_event in existing_events:
                    if existing_event['id'] == event['id']:  # Skip self
                        continue
                    
                    existing_start = existing_event['scheduled_start']
                    existing_end = existing_event['scheduled_end']
                    
                    if (alt_start < existing_end and alt_end > existing_start):
                        has_conflict = True
                        break
                
                if not has_conflict:
                    logger.info(f"RESCHEDULE_ALTERNATIVE_FOUND: Using alternative slot {i+1} for '{event['title']}'")
                    return {
                        'start_time': alt_start,
                        'end_time': alt_end,
                        'fit_score': alt_slot.get('fit_score', 0.5),
                        'priority_score': alt_slot.get('priority_score', 0.5),
                        'full_score': alt_slot.get('score', 0.5),
                        'source': 'alternative_slot'
                    }
                    
            except (ValueError, KeyError) as e:
                logger.warning(f"RESCHEDULE_ALTERNATIVE_ERROR: Invalid alternative slot data: {e}")
                continue
        
        logger.info(f"RESCHEDULE_NO_VALID_ALTERNATIVES: All {len(alternative_slots)} alternatives have conflicts")
        return None

    async def find_optimal_slot(self, user_id: str, 
                              task_type: TaskType,
                              duration: float,
                              available_periods: List[Tuple[datetime, datetime]],
                              existing_events: List[Dict],
                              importance_score: float = 0.5,
                              deadline: Optional[datetime] = None) -> Optional[Dict]:
        """Find optimal time slot with smart conflict resolution and rescheduling"""
        
        # Calculate priority_score from importance_score and deadline
        priority_score = self._calculate_priority_score(importance_score, deadline)
        
        logger.info(f"SLOT_SEARCH: Finding {duration}h slot for '{task_type.task_type}' with importance {importance_score:.2f}, priority {priority_score:.2f}")
        
        # 🔧 FIX: Get user energy pattern once to avoid repeated DB queries
        user_energy_pattern = await self._get_user_energy_pattern(user_id)
        
        # Separate fixed and movable events
        fixed_events = [e for e in existing_events if not e.get('task_type_id')]
        movable_events = [e for e in existing_events if e.get('task_type_id')]
        
        logger.info(f"EVENTS_BREAKDOWN: {len(fixed_events)} fixed events, {len(movable_events)} movable events")
        
        # Filter periods to exclude fixed events
        available_periods = self.filter_available_periods(available_periods, fixed_events)
        
        if not available_periods:
            logger.warning("SLOT_SEARCH_FAILED: No time periods available after filtering fixed events")
            return None
        
        slot_candidates = []
        
        # Search within filtered periods
        for period_start, period_end in available_periods:
            current_time = period_start.replace(minute=0, second=0, microsecond=0)
            
            if current_time < period_start:
                current_time = period_start
            
            while current_time + timedelta(hours=duration) <= period_end:
                # Calculate fit_score (50% habit + 50% energy) - pass pre-fetched energy pattern
                fit_score = await self.calculate_fit_score(
                    current_time, duration, task_type, user_id, user_energy_pattern
                )
                
                # Check conflicts with movable events only
                conflict_result = self.resolve_slot_conflicts(
                    current_time, duration, fit_score, priority_score, movable_events
                )
                
                if conflict_result['available']:
                    full_score = self.calculate_full_score(fit_score, priority_score)
                    slot_candidates.append({
                        'start_time': current_time,
                        'end_time': current_time + timedelta(hours=duration),
                        'fit_score': fit_score,
                        'priority_score': priority_score,
                        'full_score': full_score,
                        'action': conflict_result['action'],
                        'displaced_events': conflict_result.get('displaced_events', [])
                    })
                
                current_time += timedelta(minutes=30)
        
        if not slot_candidates:
            logger.warning("SLOT_SEARCH_FAILED: No suitable slots found")
            return None
        
        # Sort by full_score
        best_slots = sorted(slot_candidates, key=lambda x: x['full_score'], reverse=True)
        best_slot = best_slots[0]
        
        # Handle displacement and rescheduling
        if best_slot['action'] == 'displace':
            logger.info(f"SLOT_SELECTED_WITH_DISPLACEMENT: Displacing {len(best_slot['displaced_events'])} events")
            
            # Reschedule displaced events
            for displaced_event in best_slot['displaced_events']:
                try:
                    # First try to use alternative slots
                    alternative_slot = await self.reschedule_using_alternatives(displaced_event, existing_events)
                    
                    if alternative_slot:
                        # Update event with alternative slot
                        await self.db_service.update_event_time(
                            displaced_event['id'],
                            alternative_slot['start_time'],
                            alternative_slot['end_time']
                        )
                        logger.info(f"RESCHEDULED_ALTERNATIVE: Event '{displaced_event['title']}' moved to alternative slot")
                    else:
                        # Delete and reschedule using full search
                        await self.db_service.delete_event(displaced_event['id'])
                        
                        # Reschedule it (recursive call)
                        await self.schedule_with_pattern(
                            user_id=displaced_event['user_id'],
                            summary=displaced_event['title'],
                            description=displaced_event.get('description'),
                            duration=(displaced_event['scheduled_end'] - displaced_event['scheduled_start']).total_seconds() / 3600,
                            importance_score=displaced_event.get('priority_score', 0.5),
                            available_periods=available_periods
                        )
                        logger.info(f"RESCHEDULED_FULL_SEARCH: Event '{displaced_event['title']}' rescheduled via full search")
                        
                except Exception as e:
                    logger.error(f"RESCHEDULE_FAILED: Could not reschedule event '{displaced_event['title']}': {e}")
        
        logger.info(f"SLOT_SELECTED: {best_slot['start_time'].strftime('%m/%d %H:%M')} to {best_slot['end_time'].strftime('%H:%M')} (full_score: {best_slot['full_score']:.3f})")
        
        return {
            'optimal_slot': best_slot,
            'alternatives': best_slots[1:3],
            'total_candidates': len(slot_candidates),
            'displacement_required': best_slot['action'] == 'displace'
        }

    def _calculate_priority_score(self, importance_score: float, deadline: Optional[datetime]) -> float: # 根据重要性和截止日期计算优先级
        if deadline is None:
            return importance_score
        
        now = datetime.now()
        time_until_deadline = (deadline - now).total_seconds() / 3600  # 转换为小时
        
        if time_until_deadline <= 0:  # 已过期
            return min(1.0, importance_score * 1.5)  # 增加紧迫性
        elif time_until_deadline <= 24:  # 24小时内
            urgency_boost = 1.3
        elif time_until_deadline <= 72:  # 3天内
            urgency_boost = 1.1
        else:  # 3天以上
            urgency_boost = 1.0
        
        return min(1.0, importance_score * urgency_boost)

    def calculate_full_score(self, fit_score: float, priority_score: float) -> float:
        """Calculate full score using formula: (priority^1.5) * (fit^0.5)"""
        return (priority_score ** 1.5) * (fit_score ** 0.5)

    def resolve_slot_conflicts(self, start_time: datetime, 
                             duration: float, 
                             new_task_fit_score: float,
                             new_task_priority_score: float,
                             existing_events: List[Dict]) -> Dict:
        """Resolve slot conflicts by comparing full scores with displacement penalty"""
        end_time = start_time + timedelta(hours=duration)
        
        conflicting_events = []
        immovable_conflicts = []
        
        for event in existing_events:
            event_start = event['scheduled_start']
            event_end = event['scheduled_end']
            
            if (start_time < event_end and end_time > event_start):
                # Check if event is movable (has task_type_id) or immovable
                if event.get('task_type_id'):
                    conflicting_events.append(event)
                else:
                    immovable_conflicts.append(event)
        
        # If any immovable conflicts, cannot use this slot
        if immovable_conflicts:
            return {'available': False, 'action': 'skip', 'reason': 'immovable_conflict'}
        
        if not conflicting_events:
            return {'available': True, 'action': 'schedule', 'displaced_events': []}
        
        # Calculate full scores with displacement penalty
        new_task_full_score = self.calculate_full_score(new_task_fit_score, new_task_priority_score)
        total_conflict_full_score = 0
        displacement_penalty = 0.1  # Penalty for moving existing tasks
        
        for event in conflicting_events:
            stored_fit_score = event.get('fit_score', 0.5)
            stored_priority_score = event.get('priority_score', 0.5)
            event_full_score = self.calculate_full_score(stored_fit_score, stored_priority_score)
            penalized_full_score = event_full_score + displacement_penalty
            total_conflict_full_score += penalized_full_score
        
        # Compare full scores and decide
        if new_task_full_score > total_conflict_full_score:
            logger.info(f"CONFLICT_RESOLUTION: New task full_score {new_task_full_score:.3f} > conflicting events {total_conflict_full_score:.3f} (with penalty), displacing {len(conflicting_events)} events")
            return {'available': True, 'action': 'displace', 'displaced_events': conflicting_events}
        else:
            return {'available': False, 'action': 'skip', 'reason': 'insufficient_score'}

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

    def _parse_available_periods_string(self, periods_str: str) -> List[Tuple[datetime, datetime]]:
        """Parse available_periods string format into list of datetime tuples
        
        Format: "start1,end1;start2,end2" where dates are ISO 8601
        Example: "2024-01-15T09:00:00,2024-01-15T17:00:00;2024-01-16T09:00:00,2024-01-16T17:00:00"
        """
        try:
            periods = []
            if not periods_str.strip():
                return periods
                
            # Split by semicolon to get individual periods
            period_parts = periods_str.split(';')
            
            for period_part in period_parts:
                period_part = period_part.strip()
                if not period_part:
                    continue
                    
                # Split by comma to get start and end
                start_str, end_str = period_part.split(',')
                start_str = start_str.strip()
                end_str = end_str.strip()
                
                # Parse ISO 8601 format
                start_dt = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                
                periods.append((start_dt, end_dt))
                
            logger.info(f"📅 Parsed {len(periods)} available periods from string")
            return periods
            
        except Exception as e:
            logger.error(f"📅 Error parsing available_periods string '{periods_str}': {e}")
            return []

    def _setup_time_periods(self, available_periods: Optional[List[Tuple[datetime, datetime]]] = None) -> List[Tuple[datetime, datetime]]:
        """Set up time periods for scheduling"""
        if available_periods is None:
            # Default: search next 7 days starting from tomorrow morning at 6 AM
            now = datetime.now()
            tomorrow_morning = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
            
            start_date = tomorrow_morning
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
                .select("id, user_id, task_type_id, title, description, scheduled_start, scheduled_end, fit_score, priority_score, alternative_slots") \
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
                    "fit_score": event_data.get("fit_score", 0.5),
                    "priority_score": event_data.get("priority_score", 0.5),
                    "alternative_slots": event_data.get("alternative_slots", [])
                })
            
            print(f"📅 Found {len(existing_events)} existing events in time range")
            return existing_events
            
        except Exception as e:
            print(f"⚠️ Could not fetch existing events: {e}")
            return []  # Return empty list on error

    
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

    # ============================================================================ 
    # COLLISION DETECTION FOR DIRECT SCHEDULING
    # ============================================================================
    

    def _find_conflicting_events(self, start_time: datetime, end_time: datetime, existing_events: List[Dict]) -> List[Dict]: # 查找冲突的事件
        conflicting = []
        for event in existing_events:
            if (start_time < event['scheduled_end'] and end_time > event['scheduled_start']):
                conflicting.append(event)
        return conflicting
    
    async def _displace_conflicting_events(self, conflicting_events: List[Dict]): # 移动冲突的可移动事件
        for event in conflicting_events:
            try:
                # 尝试使用备选时间段
                alternative_slot = await self.reschedule_using_alternatives(event, [])
                
                if alternative_slot:
                    await self.db_service.update_event_time(
                        event['id'],
                        alternative_slot['start_time'],
                        alternative_slot['end_time']
                    )
                    logger.info(f"DISPLACED: Moved '{event['title']}' to alternative slot")
                else:
                    # 删除并重新调度
                    await self.db_service.delete_event(event['id'])
                    logger.info(f"DISPLACED: Deleted '{event['title']}' for rescheduling")
                    
                    duration = (event['scheduled_end'] - event['scheduled_start']).total_seconds() / 3600
                    await self.schedule_with_pattern(
                        user_id=event['user_id'],
                        summary=event['title'],
                        description=event.get('description'),
                        duration=duration,
                        importance_score=0.5
                    )
                    logger.info(f"DISPLACED: Rescheduled '{event['title']}'")
                    
            except Exception as e:
                logger.error(f"DISPLACEMENT_ERROR: Failed to displace event '{event['title']}': {e}")
        
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