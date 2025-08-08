"""
Structured Models for Instructor-based LLM Outputs
Provides Pydantic models for reliable structured data extraction
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class MemoryCategory(str, Enum):
    """Memory categories for user insights"""
    PRIORITY = "priority"
    HABIT = "habit" 
    ENERGY = "energy"
    CONTEXT = "context"


class MemoryCategorization(BaseModel):
    """Structured output for memory categorization"""
    priority_insights: List[str] = Field(
        default_factory=list,
        description="Task importance signals, urgency markers, deadlines, critical needs"
    )
    habit_insights: List[str] = Field(
        default_factory=list,
        description="Time preferences, routine patterns, scheduling habits, recurring behaviors"
    )
    energy_insights: List[str] = Field(
        default_factory=list,
        description="Energy levels, focus patterns, productivity observations, performance notes"
    )
    context_insights: List[str] = Field(
        default_factory=list,
        description="Goals, feelings, life situation, general context, other relevant information"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in categorization (0.0-1.0)"
    )
    
    @field_validator('confidence')
    def validate_confidence(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class TaskPatternAnalysis(BaseModel):
    """Structured output for LLM task pattern analysis in task_type_service"""
    time_patterns: str = Field(
        description="Compact time patterns string format: 'days:hour_start-hour_end:preference_score,days:hour_start-hour_end:preference_score'"
    )
    base_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Single confidence level (0.0-1.0) applied to all time slots"
    )
    cognitive_load: float = Field(
        ge=0.0, le=1.0,
        description="How mentally demanding this task is (0.0=easy/relaxing, 1.0=intense focus)"
    )
    typical_duration: float = Field(
        ge=0.1, le=24.0,
        description="How long this type of task usually takes in hours"
    )
    importance_score: float = Field(
        ge=0.0, le=1.0,
        description="General priority level for this task type (0.0=low priority, 1.0=critical)"
    )
    recovery_hours: float = Field(
        ge=0.0, le=4.0,
        description="Buffer time needed after completion in hours"
    )
    analysis_reasoning: str = Field(
        description="Brief explanation of the behavioral pattern analysis"
    )
    
    @field_validator('base_confidence', 'cognitive_load', 'importance_score')
    def validate_scores(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Scores must be between 0.0 and 1.0')
        return v
    
    @field_validator('typical_duration')
    def validate_duration(cls, v):
        if v <= 0:
            raise ValueError('Duration must be positive')
        return v 