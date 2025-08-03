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


class User(BaseModel):
    id: UUID
    email: str
    role: UserRole
    timezone: str = "UTC"
    
    # Weekly energy pattern (168 elements: 24 hours × 7 days)
    # Index formula: day_of_week * 24 + hour (0=Sunday 00:00, 167=Saturday 23:59)
    weekly_energy_pattern: List[float] = Field(default_factory=lambda: [0.5] * 168)
    
    created_at: datetime


class TaskType(BaseModel):
    """Tier 2: Task Types with learned patterns"""
    id: UUID
    user_id: UUID
    task_type: str
    description: Optional[str] = None  # Optional description for better embedding
    
    # 168-hour weekly array (24 hours × 7 days)
    # Index formula: day_of_week * 24 + hour (0=Sunday 00:00, 167=Saturday 23:59)
    weekly_habit_scores: List[float] = Field(default_factory=lambda: [0.0] * 168)
    
    # 7x24 confidence matrix (confidence for each time slot)
    slot_confidence: List[List[float]] = Field(default_factory=lambda: [[0.0]*24 for _ in range(7)])
    
    # Completion tracking
    completion_count: int = 0        # Total number of times this task type has been completed
    completions_since_last_update: int = 0  # Completions since last mem0 update (resets to 0 after update)
    last_mem0_update: Optional[datetime] = None  # Track when last updated from mem0 insights
    
    # Task characteristics
    typical_duration: float = 1.0    # Average duration in hours
    importance_score: float = 0.5    # Learned importance from mem0 (0-1)
    recovery_hours: float = 0.5      # Buffer time needed after this task
    cognitive_load: float = 0.5      # Task mental difficulty (0-1)
    
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
    description: Optional[str] = None


class ScheduleEventRequest(BaseModel):
    title: str
    description: Optional[str] = None
    duration: float = 1.0  # hours
    preferred_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    importance_score: float = 0.5  # 0.0-1.0, default moderate importance


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


def initialize_neutral_weekly_habit_array() -> List[float]:
    """Initialize zero 168-hour weekly habit array for new task types"""
    weekly_habit_scores = [0.0] * 168  # Start from zero for all week
    return weekly_habit_scores


def get_weekly_index(day_of_week: int, hour: int) -> int:
    """Convert day of week (0=Sunday) and hour (0-23) to weekly array index"""
    return day_of_week * 24 + hour


def get_day_hour_from_index(index: int) -> tuple:
    """Convert weekly array index back to (day_of_week, hour)"""
    day_of_week = index // 24
    hour = index % 24
    return day_of_week, hour


def initialize_neutral_weekly_energy() -> List[float]:
    """Initialize neutral 168-hour weekly energy pattern for new users"""
    weekly_energy = [0.5] * 168  # Neutral energy for all 168 hours (24 * 7)
    return weekly_energy

def initialize_slot_confidence() -> List[List[float]]:
    """Initialize slot confidence matrix (7x24 = 7 days, 24 hours)"""
    return [[0.0]*24 for _ in range(7)]