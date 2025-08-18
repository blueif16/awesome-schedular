"""
Database Service - Event Management
Handles event creation and database operations
"""

import uuid
from datetime import datetime
from typing import Optional
from supabase import Client


class DatabaseService:
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    async def create_event(self,
                          user_id: str,
                          title: str,
                          description: Optional[str],
                          scheduled_start: datetime,
                          scheduled_end: datetime,
                          task_type_id: Optional[str] = None,
                          item_type: str = "task",
                          auto_schedule: bool = True,
                          alternative_slots: Optional[list] = None,
                          timezone: str = "UTC",
                          location: Optional[str] = None,
                          importance_score: float = 0.5,
                          deadline: Optional[datetime] = None) -> str:
        """
        Create a new event in the database with unified schema
        
        Args:
            user_id: User identifier
            title: Event title
            description: Optional event description  
            scheduled_start: Event start time
            scheduled_end: Event end time
            task_type_id: Optional task type ID for pattern tracking
            item_type: "task" or "event" to distinguish type
            auto_schedule: Whether this event can be moved automatically
            alternative_slots: List of alternative time slots
            timezone: Event timezone
            location: Optional event location
            importance_score: Priority score (0.0-1.0)
            deadline: Optional deadline for urgency
            
        Returns:
            str: Event ID if successful
        """
        try:
            event_data = {
                "user_id": user_id,
                "title": title,
                "description": description,
                "scheduled_start": scheduled_start.isoformat(),
                "scheduled_end": scheduled_end.isoformat(),
                "item_type": item_type,
                "auto_schedule": auto_schedule,
                "alternative_slots": alternative_slots or [],
                "timezone": timezone,
                "location": location,
                "importance_score": importance_score,
                "completed": False
            }
            
            # Add optional fields if provided
            if task_type_id:
                event_data["task_type_id"] = task_type_id
            if deadline:
                event_data["deadline"] = deadline.isoformat()
            
            print(f"💾 Creating event: '{title}' from {scheduled_start.strftime('%m/%d %H:%M')} to {scheduled_end.strftime('%m/%d %H:%M')}")
            
            result = self.supabase.table("events").insert(event_data).execute()
            
            if result.data:
                event_id = result.data[0]["id"]
                print(f"✅ Event created successfully with ID: {event_id}")
                return event_id
            else:
                raise Exception("No data returned from event creation")
                
        except Exception as e:
            print(f"❌ Error creating event: {e}")
            raise



