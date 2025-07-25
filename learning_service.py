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
    def update_hourly_score(current_score: float, 
                           success: bool, 
                           confidence: float,
                           base_learning_rate: float = 0.3) -> tuple:
        """
        Update hourly score based on task completion
        Returns: (new_score, new_confidence)
        """
        # Adaptive learning rate - learn faster when less confident
        learning_rate = base_learning_rate * (1 - confidence)
        
        # Success reinforces, failure reduces
        signal = 0.9 if success else 0.1
        
        # Weighted update: current_score * (1 - rate) + signal * rate
        new_score = current_score * (1 - learning_rate) + signal * learning_rate
        
        # Increase confidence gradually (cap at 0.95)
        new_confidence = min(0.95, confidence + 0.05)
        
        return new_score, new_confidence
    
    async def update_task_type_patterns(self, task_type_id: str, 
                                      completion_hour: int,
                                      success: bool, 
                                      energy_after: float,
                                      success_rating: float = None):
        """Update all patterns for a task type after completion"""
        
        # Get current task type
        task_type = await self.task_type_service.get_task_type(task_type_id)
        if not task_type:
            print(f"Task type {task_type_id} not found")
            return
        
        # Determine success from success_rating if provided
        if success_rating is not None:
            success = success_rating > 0.7
        
        # Get current arrays
        hourly_scores = task_type.hourly_scores.copy()
        confidence_scores = task_type.confidence_scores.copy()
        performance_scores = task_type.performance_by_hour.copy()
        
        # Update hourly preference for this hour
        new_score, new_confidence = self.update_hourly_score(
            hourly_scores[completion_hour],
            success,
            confidence_scores[completion_hour]
        )
        
        hourly_scores[completion_hour] = new_score
        confidence_scores[completion_hour] = new_confidence
        
        # Update performance (energy) for this hour
        if energy_after is not None:
            # Weighted average with previous performance
            current_performance = performance_scores[completion_hour]
            current_confidence = confidence_scores[completion_hour] - 0.05  # Previous confidence
            
            # Weight new energy data based on confidence
            weight = 0.3 if current_confidence > 0.5 else 0.5
            performance_scores[completion_hour] = (
                current_performance * (1 - weight) + energy_after * weight
            )
        
        # Save updated arrays back to database
        await self.task_type_service.update_task_type_arrays(
            str(task_type.id),
            hourly_scores,
            confidence_scores,
            performance_scores
        )
        
        print(f"✅ Updated patterns for '{task_type.task_type}' at hour {completion_hour}")
        print(f"   Preference: {hourly_scores[completion_hour]:.2f} (confidence: {confidence_scores[completion_hour]:.2f})")
        print(f"   Energy: {performance_scores[completion_hour]:.2f}")
    
    def analyze_patterns(self, hourly_scores: List[float], 
                        confidence_scores: List[float]) -> dict:
        """Analyze learned patterns to find insights"""
        
        # Find peak hours (high score + high confidence)
        peak_hours = []
        for hour in range(24):
            if hourly_scores[hour] > 0.7 and confidence_scores[hour] > 0.5:
                peak_hours.append(hour)
        
        # Find low hours
        low_hours = []
        for hour in range(24):
            if hourly_scores[hour] < 0.3 and confidence_scores[hour] > 0.5:
                low_hours.append(hour)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Identify time blocks
        morning_block = sum(hourly_scores[6:12]) / 6  # 6 AM - 12 PM
        afternoon_block = sum(hourly_scores[12:18]) / 6  # 12 PM - 6 PM
        evening_block = sum(hourly_scores[18:24]) / 6  # 6 PM - 12 AM
        
        return {
            "peak_hours": peak_hours,
            "low_hours": low_hours,
            "avg_confidence": avg_confidence,
            "morning_preference": morning_block,
            "afternoon_preference": afternoon_block,
            "evening_preference": evening_block,
            "best_time_block": max(
                [("morning", morning_block), ("afternoon", afternoon_block), ("evening", evening_block)],
                key=lambda x: x[1]
            )[0]
        }
    
    def generate_pattern_insights(self, task_type_name: str, 
                                analysis: dict) -> List[str]:
        """Generate human-readable insights from pattern analysis"""
        insights = []
        
        if analysis["peak_hours"]:
            hours_str = ", ".join([f"{h}:00" for h in analysis["peak_hours"]])
            insights.append(f"🎯 Best times for '{task_type_name}': {hours_str}")
        
        if analysis["low_hours"]:
            hours_str = ", ".join([f"{h}:00" for h in analysis["low_hours"]])
            insights.append(f"❌ Avoid '{task_type_name}' at: {hours_str}")
        
        best_block = analysis["best_time_block"]
        insights.append(f"⏰ You prefer '{task_type_name}' in the {best_block}")
        
        confidence = analysis["avg_confidence"]
        if confidence > 0.7:
            insights.append(f"📊 High confidence in these patterns ({confidence:.1%})")
        elif confidence < 0.3:
            insights.append(f"📊 Need more data to learn patterns ({confidence:.1%})")
        
        return insights
    
    async def get_user_learning_stats(self, user_id: str) -> dict:
        """Get learning statistics for a user"""
        task_types = await self.task_type_service.get_user_task_types(user_id)
        
        if not task_types:
            return {
                "total_task_types": 0,
                "avg_confidence": 0,
                "highly_confident_hours": 0,
                "insights": []
            }
        
        total_confidence = 0
        highly_confident_hours = 0
        all_insights = []
        
        for task_type in task_types:
            # Analyze this task type
            analysis = self.analyze_patterns(
                task_type.hourly_scores,
                task_type.confidence_scores
            )
            
            total_confidence += analysis["avg_confidence"]
            highly_confident_hours += len(analysis["peak_hours"])
            
            # Generate insights
            insights = self.generate_pattern_insights(task_type.task_type, analysis)
            all_insights.extend(insights)
        
        avg_confidence = total_confidence / len(task_types)
        
        return {
            "total_task_types": len(task_types),
            "avg_confidence": avg_confidence,
            "highly_confident_hours": highly_confident_hours,
            "insights": all_insights
        } 