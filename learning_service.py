"""
Learning Service - Pattern Learning Implementation
Updates Tier 2 patterns based on task completion feedback
"""

import math
from typing import List, Optional
from task_type_service import TaskTypeService


class LearningService:
    def __init__(self, task_type_service: TaskTypeService):
        self.task_type_service = task_type_service
    
    @staticmethod
    def update_habit_score(current_score: float, 
                          success: bool, 
                          completion_count: int,
                          base_learning_rate: float = 0.3) -> float:
        """
        Update habit score based on task completion with regularization
        Learning rate decreases as completion count increases (more stable with more data)
        """
        # Adaptive learning rate - learn faster with fewer completions
        learning_rate = base_learning_rate / (1 + completion_count * 0.1)
        
        # Success reinforces, failure reduces
        signal = 0.9 if success else 0.1
        
        # Weighted update: current_score * (1 - rate) + signal * rate
        new_score = current_score * (1 - learning_rate) + signal * learning_rate
        
        # Regularization: prevent extreme values, pull toward neutral (0.5)
        regularization_strength = 0.05
        if new_score > 0.8:
            new_score = new_score * (1 - regularization_strength) + 0.5 * regularization_strength
        elif new_score < 0.2:
            new_score = new_score * (1 - regularization_strength) + 0.5 * regularization_strength
        
        return max(0.0, min(1.0, new_score))  # Clamp between 0 and 1
    
    async def update_task_type_patterns(self, task_type_id: str, 
                                      completion_hour: int,
                                      success: bool, 
                                      energy_after: float = None,
                                      success_rating: float = None,
                                      user_id: str = None,
                                      hybrid_learning_service = None):
        """Update habit patterns for a task type after completion"""
        
        # Get current task type
        task_type = await self.task_type_service.get_task_type(task_type_id)
        if not task_type:
            print(f"Task type {task_type_id} not found")
            return
        
        # Determine success from success_rating if provided
        if success_rating is not None:
            success = success_rating > 0.7
        
        # Get current weekly habit scores and completion count
        weekly_habit_scores = task_type.weekly_habit_scores.copy()
        current_completion_count = task_type.completion_count
        
        # Convert completion hour to weekly index (assumes current day is today)
        from datetime import datetime
        current_day = datetime.now().weekday()  # 0=Monday, but we need 0=Sunday
        day_of_week = (current_day + 1) % 7  # Convert to 0=Sunday format
        
        from models import get_weekly_index
        weekly_index = get_weekly_index(day_of_week, completion_hour)
        
        # Update habit score for this time slot
        new_habit_score = self.update_habit_score(
            weekly_habit_scores[weekly_index],
            success,
            current_completion_count
        )
        
        weekly_habit_scores[weekly_index] = new_habit_score
        
        # Increment completion count
        new_completion_count = current_completion_count + 1
        
        # Save updated weekly habit scores and completion count back to database
        await self.task_type_service.update_task_type_habits(
            str(task_type.id),
            weekly_habit_scores,
            new_completion_count
        )
        
        day_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][day_of_week]
        print(f"✅ Updated patterns for '{task_type.task_type}' at {day_name} {completion_hour:02d}:00")
        print(f"   Habit Score: {weekly_habit_scores[weekly_index]:.2f}")
        print(f"   Total Completions: {new_completion_count}")
        if energy_after is not None:
            print(f"   Energy After: {energy_after:.2f}")
        
        # Handle completion tracking and potential mem0 update via hybrid learning service
        if hybrid_learning_service and user_id:
            await hybrid_learning_service.handle_task_completion(
                user_id, 
                task_type_id, 
                day_of_week,  # day_of_week for slot confidence
                completion_hour, 
                success
            )
        
        # TODO: Store energy_after in user's weekly_energy_pattern (not implemented yet)

    # Update energy pattern every N completions across all task types
    # Analyze energy trends over time

    def update_energy_pattern_batch(user_id: str):
        """
        Update user energy pattern every 10-15 total completions
        Analyze recent energy feedback across all task types
        """
        # Collect last 20 completions with energy_after data
        # Group by time slots
        # Update energy pattern based on trends
    
    def analyze_weekly_patterns(self, weekly_habit_scores: List[float], 
                               completion_count: int) -> dict:
        """Analyze learned weekly patterns to find insights"""
        from models import get_day_hour_from_index
        
        if len(weekly_habit_scores) != 168:
            return {"error": "Invalid weekly habit scores array"}
        
        # Find peak time slots (high scores)
        peak_slots = []
        low_slots = []
        
        for i, score in enumerate(weekly_habit_scores):
            day_of_week, hour = get_day_hour_from_index(i)
            day_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][day_of_week]
            
            # Only consider slots with reasonable confidence (based on completion count)
            confidence_threshold = min(0.8, completion_count * 0.1)  # More completions = higher confidence
            
            if score > 0.7 and completion_count > 5:  # Need some data to trust
                peak_slots.append(f"{day_name} {hour:02d}:00")
            elif score < 0.3 and completion_count > 5:
                low_slots.append(f"{day_name} {hour:02d}:00")
        
        # Calculate day-of-week preferences
        day_preferences = {}
        day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        for day in range(7):
            day_start = day * 24
            day_end = day_start + 24
            day_scores = weekly_habit_scores[day_start:day_end]
            day_preferences[day_names[day]] = sum(day_scores) / len(day_scores)
        
        # Find best and worst days
        best_day = max(day_preferences.items(), key=lambda x: x[1])
        worst_day = min(day_preferences.items(), key=lambda x: x[1])
        
        # Calculate time-of-day preferences (averaged across all days)
        hourly_averages = []
        for hour in range(24):
            hour_scores = []
            for day in range(7):
                index = day * 24 + hour
                hour_scores.append(weekly_habit_scores[index])
            hourly_averages.append(sum(hour_scores) / len(hour_scores))
        
        # Identify best time blocks
        morning_avg = sum(hourly_averages[6:12]) / 6    # 6 AM - 12 PM
        afternoon_avg = sum(hourly_averages[12:18]) / 6  # 12 PM - 6 PM
        evening_avg = sum(hourly_averages[18:24]) / 6    # 6 PM - 12 AM
        
        return {
            "peak_slots": peak_slots[:10],  # Top 10 peak slots
            "low_slots": low_slots[:10],    # Top 10 low slots
            "completion_count": completion_count,
            "day_preferences": day_preferences,
            "best_day": best_day,
            "worst_day": worst_day,
            "morning_preference": morning_avg,
            "afternoon_preference": afternoon_avg,
            "evening_preference": evening_avg,
            "best_time_block": max(
                [("morning", morning_avg), ("afternoon", afternoon_avg), ("evening", evening_avg)],
                key=lambda x: x[1]
            )[0],
            "hourly_averages": hourly_averages
        }
    
    def generate_weekly_pattern_insights(self, task_type_name: str, 
                                        analysis: dict) -> List[str]:
        """Generate human-readable insights from weekly pattern analysis"""
        insights = []
        
        if "error" in analysis:
            insights.append(f"⚠️ {analysis['error']}")
            return insights
        
        completion_count = analysis["completion_count"]
        
        # Data quality insight
        if completion_count < 5:
            insights.append(f"📊 Need more data to learn patterns ({completion_count} completions)")
            return insights
        elif completion_count < 15:
            insights.append(f"📊 Learning patterns ({completion_count} completions)")
        else:
            insights.append(f"📊 Strong pattern data ({completion_count} completions)")
        
        # Peak time slots
        if analysis["peak_slots"]:
            peak_times = ", ".join(analysis["peak_slots"][:5])  # Show top 5
            insights.append(f"🎯 Best times for '{task_type_name}': {peak_times}")
        
        # Low time slots
        if analysis["low_slots"]:
            low_times = ", ".join(analysis["low_slots"][:3])   # Show top 3 to avoid
            insights.append(f"❌ Avoid '{task_type_name}' at: {low_times}")
        
        # Day preferences
        best_day, best_score = analysis["best_day"]
        worst_day, worst_score = analysis["worst_day"]
        
        if best_score > 0.6:
            insights.append(f"📅 '{task_type_name}' works best on {best_day}")
        
        if worst_score < 0.4 and best_score - worst_score > 0.2:
            insights.append(f"📅 '{task_type_name}' struggles on {worst_day}")
        
        # Time of day preferences
        best_block = analysis["best_time_block"]
        insights.append(f"⏰ You prefer '{task_type_name}' in the {best_block}")
        
        # Specific patterns
        morning_pref = analysis["morning_preference"]
        afternoon_pref = analysis["afternoon_preference"]
        evening_pref = analysis["evening_preference"]
        
        if morning_pref > afternoon_pref * 1.3 and morning_pref > evening_pref * 1.3:
            insights.append(f"🌅 Strong morning preference for '{task_type_name}'")
        elif evening_pref > morning_pref * 1.3 and evening_pref > afternoon_pref * 1.3:
            insights.append(f"🌙 Evening owl pattern for '{task_type_name}'")
        elif afternoon_pref < morning_pref * 0.7 and afternoon_pref < evening_pref * 0.7:
            insights.append(f"😴 Afternoon energy dip affects '{task_type_name}'")
        
        return insights
    
    async def get_user_learning_stats(self, user_id: str) -> dict:
        """Get learning statistics for a user"""
        task_types = await self.task_type_service.get_user_task_types(user_id)
        
        if not task_types:
            return {
                "total_task_types": 0,
                "total_completions": 0,
                "peak_time_slots": 0,
                "insights": []
            }
        
        total_completions = 0
        peak_time_slots = 0
        all_insights = []
        
        for task_type in task_types:
            # Analyze this task type's weekly patterns
            analysis = self.analyze_weekly_patterns(
                task_type.weekly_habit_scores,
                task_type.completion_count
            )
            
            total_completions += task_type.completion_count
            if "peak_slots" in analysis:
                peak_time_slots += len(analysis["peak_slots"])
            
            # Generate insights
            insights = self.generate_weekly_pattern_insights(task_type.task_type, analysis)
            all_insights.extend(insights)
        
        return {
            "total_task_types": len(task_types),
            "total_completions": total_completions,
            "peak_time_slots": peak_time_slots,
            "insights": all_insights
        } 