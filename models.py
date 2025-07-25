"""
Smart Scheduler: Pydantic Models
Data structures for the three-tier architecture
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from enum import Enum


class UserRole(str, Enum):
    STUDENT = "student"
    PM = "pm" 
    DEVELOPER = "developer"
    EXECUTIVE = "executive"


class TaskCategory(str, Enum):
    FOCUSED = "focused"          # Deep work, coding, studying
    COLLABORATIVE = "collaborative"  # Meetings, team work
    ADMINISTRATIVE = "administrative"  # Email, planning, admin tasks


class User(BaseModel):
    id: UUID
    email: str
    role: UserRole
    timezone: str = "UTC"
    created_at: datetime


class TaskType(BaseModel):
    """Tier 2: Task Types with learned patterns"""
    id: UUID
    user_id: UUID
    task_type: str
    category: TaskCategory
    
    # 24-hour arrays (indices 0-23 for hours 00:00-23:59)
    hourly_scores: List[float] = Field(default_factory=lambda: [0.5] * 24)
    confidence_scores: List[float] = Field(default_factory=lambda: [0.1] * 24)
    performance_by_hour: List[float] = Field(default_factory=lambda: [0.5] * 24)
    
    # Task characteristics
    cognitive_load: float = 0.5      # 0-1: How mentally demanding
    recovery_hours: float = 0.5      # Hours needed to recover
    typical_duration: float = 1.0    # Average duration in hours
    importance_score: float = 0.5    # Learned importance
    
    # Vector embedding for similarity
    embedding: Optional[List[float]] = None
    
    created_at: datetime
    updated_at: datetime


class Event(BaseModel):
    """Tier 1: Individual events/tasks"""
    id: UUID
    user_id: UUID
    task_type_id: UUID
    
    title: str
    description: Optional[str] = None
    scheduled_start: datetime
    scheduled_end: datetime
    deadline: Optional[datetime] = None
    
    # Completion tracking for learning
    completed: bool = False
    success_rating: Optional[float] = None      # 0-1: How well did it go?
    energy_after: Optional[float] = None        # 0-1: Energy after completion
    energy_before: Optional[float] = None       # 0-1: Energy before start
    perceived_difficulty: Optional[float] = None  # 0-1: How hard was it?
    
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    calculated_priority: float = 0.5
    
    created_at: datetime


# Request/Response Models

class CreateUserRequest(BaseModel):
    email: str
    role: UserRole
    timezone: str = "UTC"


class CreateTaskTypeRequest(BaseModel):
    task_type: str
    category: TaskCategory
    description: Optional[str] = None


class ScheduleEventRequest(BaseModel):
    title: str
    description: Optional[str] = None
    duration: float = 1.0  # hours
    preferred_date: Optional[datetime] = None
    deadline: Optional[datetime] = None


class CompleteEventRequest(BaseModel):
    success_rating: float = Field(ge=0, le=1)
    energy_after: float = Field(ge=0, le=1)
    energy_before: Optional[float] = Field(ge=0, le=1, default=None)
    perceived_difficulty: Optional[float] = Field(ge=0, le=1, default=None)
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None


class SchedulingResult(BaseModel):
    """Result of optimal scheduling"""
    event: Event
    optimal_slot: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    reasoning: str
    pattern_insights: Dict[str, Any]


class TaskTypeSimilarity(BaseModel):
    """Similar task type with similarity score"""
    task_type: TaskType
    similarity: float


def initialize_neutral_arrays() -> tuple:
    """Initialize neutral 24-hour arrays for new task types"""
    hourly_scores = [0.5] * 24      # Neutral preference
    confidence_scores = [0.1] * 24  # Low confidence initially  
    performance_by_hour = [0.5] * 24  # Neutral performance
    return hourly_scores, confidence_scores, performance_by_hour 