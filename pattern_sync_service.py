"""
Pattern Sync Service - Orchestrates Hybrid Learning Strategy
Manages the dual update approach: real-time behavioral + periodic LLM interpretation
"""

import asyncio
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from hybrid_learning_service import HybridLearningService, BehavioralUpdate
from task_type_service import TaskTypeService
from models import get_weekly_index


@dataclass
class PatternSyncConfig:
    """Configuration for pattern synchronization"""
    mem0_sync_interval_hours: int = 6          # How often to sync with Mem0
    max_behavioral_queue_size: int = 1000      # Max behavioral updates to queue
    learning_rate_decay: float = 0.05          # How learning rate decreases with completions
    behavioral_weight: float = 0.7             # Weight given to behavioral data
    llm_weight: float = 0.3                    # Weight given to LLM interpretations
    min_completions_for_insights: int = 5      # Min completions before generating insights
    background_sync_enabled: bool = True       # Enable background sync process


class PatternSyncService:
    """Orchestrates the hybrid learning strategy"""
    
    def __init__(self, 
                 hybrid_learning_service: HybridLearningService,
                 task_type_service: TaskTypeService,
                 memory_service,
                 config: PatternSyncConfig = None):
        
        self.hybrid_learning = hybrid_learning_service
        self.task_type_service = task_type_service
        self.memory_service = memory_service
        self.config = config or PatternSyncConfig()
        
        # Sync tracking
        self.last_sync_per_user: Dict[str, datetime] = {}
        self.behavioral_updates_pending: Dict[str, int] = {}  # user_id -> count
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            "behavioral_updates_processed": 0,
            "mem0_syncs_completed": 0,
            "patterns_updated": 0,
            "insights_generated": 0,
            "last_activity": None
        }
    
    async def start_continuous_sync(self):
        """Start the continuous pattern synchronization background process"""
        if self.is_running:
            print("⚠️ Pattern sync already running")
            return
        
        self.is_running = True
        print("🚀 Starting continuous pattern synchronization...")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._behavioral_update_processor()),
            asyncio.create_task(self._periodic_mem0_sync()),
            asyncio.create_task(self._metrics_reporter())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("📊 Pattern sync stopped")
        except Exception as e:
            print(f"❌ Error in pattern sync: {e}")
        finally:
            self.is_running = False
    
    async def stop_continuous_sync(self):
        """Stop the continuous synchronization"""
        self.is_running = False
        print("⏹️ Stopping pattern synchronization...")
    
    async def record_task_completion(self, 
                                   user_id: str,
                                   task_type_id: str, 
                                   scheduled_start: datetime,
                                   success_rating: float,
                                   energy_after: Optional[float] = None):
        """Record a task completion for behavioral learning"""
        
        # Extract day and hour from scheduled start
        day_of_week = scheduled_start.weekday()
        day_of_week = (day_of_week + 1) % 7  # Convert to 0=Sunday format
        hour = scheduled_start.hour
        
        # Create behavioral update
        update = BehavioralUpdate(
            task_type_id=task_type_id,
            day_of_week=day_of_week,
            hour=hour,
            success_rating=success_rating,
            energy_after=energy_after,
            user_id=user_id,
            timestamp=datetime.now()
        )
        
        # Queue for immediate processing
        self.hybrid_learning.queue_behavioral_update(update)
        
        # Track pending updates for this user
        self.behavioral_updates_pending[user_id] = (
            self.behavioral_updates_pending.get(user_id, 0) + 1
        )
        
        # Update metrics
        self.metrics["behavioral_updates_processed"] += 1
        self.metrics["last_activity"] = datetime.now()
        
        print(f"📝 Queued behavioral update for user {user_id}")
    
    async def force_mem0_sync(self, user_id: str) -> bool:
        """Force a Mem0 sync for a specific user"""
        try:
            await self.hybrid_learning.sync_patterns_from_mem0(user_id, force=True)
            self.last_sync_per_user[user_id] = datetime.now()
            self.behavioral_updates_pending[user_id] = 0  # Reset pending count
            
            self.metrics["mem0_syncs_completed"] += 1
            return True
            
        except Exception as e:
            print(f"❌ Failed to force sync for user {user_id}: {e}")
            return False
    
    async def get_sync_status(self, user_id: str) -> Dict:
        """Get synchronization status for a user"""
        last_sync = self.last_sync_per_user.get(user_id)
        pending_updates = self.behavioral_updates_pending.get(user_id, 0)
        
        # Check if sync is needed
        needs_sync = False
        if last_sync:
            time_since_sync = datetime.now() - last_sync
            needs_sync = time_since_sync > timedelta(hours=self.config.mem0_sync_interval_hours)
        else:
            needs_sync = True
        
        # Check if user has behavioral updates waiting
        has_pending_updates = pending_updates > 0
        
        return {
            "user_id": user_id,
            "last_sync": last_sync.isoformat() if last_sync else None,
            "pending_behavioral_updates": pending_updates,
            "needs_sync": needs_sync,
            "has_pending_updates": has_pending_updates,
            "sync_interval_hours": self.config.mem0_sync_interval_hours,
            "time_until_next_sync": (
                timedelta(hours=self.config.mem0_sync_interval_hours) - 
                (datetime.now() - last_sync) if last_sync else timedelta(0)
            ).total_seconds() / 3600
        }
    
    async def get_learning_progress(self, user_id: str) -> Dict:
        """Get learning progress for a user"""
        try:
            # Get task types and their completion counts
            task_types = await self.task_type_service.get_user_task_types(user_id)
            
            if not task_types:
                return {
                    "total_task_types": 0,
                    "total_completions": 0,
                    "learning_maturity": "new_user",
                    "insights_available": 0,
                    "patterns_learned": []
                }
            
            total_completions = sum(tt.completion_count for tt in task_types)
            mature_task_types = [tt for tt in task_types if tt.completion_count >= self.config.min_completions_for_insights]
            
            # Determine learning maturity
            if total_completions < 10:
                maturity = "learning"
            elif total_completions < 50:
                maturity = "developing"
            else:
                maturity = "mature"
            
            # Get patterns for mature task types
            patterns_learned = []
            for task_type in mature_task_types:
                analysis = self.hybrid_learning.learning_service.analyze_weekly_patterns(
                    task_type.weekly_habit_scores,
                    task_type.completion_count
                )
                
                if "peak_slots" in analysis and analysis["peak_slots"]:
                    patterns_learned.append({
                        "task_type": task_type.task_type,
                        "peak_slots": analysis["peak_slots"][:3],
                        "best_day": analysis.get("best_day", ["Unknown", 0])[0],
                        "completions": task_type.completion_count
                    })
            
            return {
                "total_task_types": len(task_types),
                "total_completions": total_completions,
                "mature_task_types": len(mature_task_types),
                "learning_maturity": maturity,
                "insights_available": len(patterns_learned),
                "patterns_learned": patterns_learned,
                "sync_status": await self.get_sync_status(user_id)
            }
            
        except Exception as e:
            print(f"Error getting learning progress: {e}")
            return {"error": str(e)}
    
    async def _behavioral_update_processor(self):
        """Background task to process behavioral updates"""
        while self.is_running:
            try:
                # The hybrid learning service handles immediate processing
                # This could be extended for batch processing if needed
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error in behavioral update processor: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _periodic_mem0_sync(self):
        """Background task for periodic Mem0 synchronization"""
        while self.is_running:
            try:
                # Get users who need syncing
                users_to_sync = await self._get_users_needing_sync()
                
                for user_id in users_to_sync:
                    if not self.is_running:  # Check if we should stop
                        break
                    
                    try:
                        await self.hybrid_learning.sync_patterns_from_mem0(user_id)
                        self.last_sync_per_user[user_id] = datetime.now()
                        self.behavioral_updates_pending[user_id] = 0
                        
                        self.metrics["mem0_syncs_completed"] += 1
                        print(f"✅ Completed Mem0 sync for user {user_id}")
                        
                    except Exception as e:
                        print(f"❌ Failed Mem0 sync for user {user_id}: {e}")
                
                # Wait for next sync cycle
                await asyncio.sleep(60 * 30)  # Check every 30 minutes
                
            except Exception as e:
                print(f"Error in periodic Mem0 sync: {e}")
                await asyncio.sleep(60 * 60)  # Wait 1 hour on error
    
    async def _metrics_reporter(self):
        """Background task to report metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(60 * 60)  # Report every hour
                
                if self.metrics["last_activity"]:
                    print(f"\n📊 Pattern Sync Metrics (Last Hour)")
                    print(f"   Behavioral Updates: {self.metrics['behavioral_updates_processed']}")
                    print(f"   Mem0 Syncs: {self.metrics['mem0_syncs_completed']}")
                    print(f"   Last Activity: {self.metrics['last_activity'].strftime('%H:%M:%S')}")
                    print(f"   Users with Pending Updates: {len(self.behavioral_updates_pending)}")
                
            except Exception as e:
                print(f"Error in metrics reporter: {e}")
    
    async def _get_users_needing_sync(self) -> List[str]:
        """Get list of users who need Mem0 sync"""
        users_needing_sync = []
        
        # Check users with pending behavioral updates
        for user_id, pending_count in self.behavioral_updates_pending.items():
            if pending_count > 0:
                # Check if enough time has passed since last sync
                last_sync = self.last_sync_per_user.get(user_id)
                if not last_sync:
                    users_needing_sync.append(user_id)
                else:
                    time_since_sync = datetime.now() - last_sync
                    if time_since_sync > timedelta(hours=self.config.mem0_sync_interval_hours):
                        users_needing_sync.append(user_id)
        
        # Also check for users who haven't synced in a while (even without pending updates)
        for user_id, last_sync in self.last_sync_per_user.items():
            time_since_sync = datetime.now() - last_sync
            if time_since_sync > timedelta(hours=self.config.mem0_sync_interval_hours * 2):
                if user_id not in users_needing_sync:
                    users_needing_sync.append(user_id)
        
        return users_needing_sync
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            **self.metrics,
            "is_running": self.is_running,
            "users_tracked": len(self.behavioral_updates_pending),
            "pending_updates": sum(self.behavioral_updates_pending.values()),
            "config": {
                "sync_interval_hours": self.config.mem0_sync_interval_hours,
                "behavioral_weight": self.config.behavioral_weight,
                "llm_weight": self.config.llm_weight
            }
        }


# Example usage and integration helpers

async def create_pattern_sync_service(supabase_client, openai_api_key: str, memory_service):
    """Factory function to create a fully configured PatternSyncService"""
    
    # Initialize core services
    task_type_service = TaskTypeService(supabase_client, openai_api_key)
    hybrid_learning = HybridLearningService(task_type_service, memory_service, openai_api_key)
    
    # Create pattern sync service
    config = PatternSyncConfig(
        mem0_sync_interval_hours=6,  # Sync every 6 hours
        behavioral_weight=0.7,       # Trust behavior more
        llm_weight=0.3              # LLM provides context
    )
    
    return PatternSyncService(
        hybrid_learning, 
        task_type_service, 
        memory_service, 
        config
    )


# Integration with existing services
class SmartSchedulerWithPatternSync:
    """Enhanced scheduler with hybrid pattern learning"""
    
    def __init__(self, pattern_sync_service: PatternSyncService):
        self.pattern_sync = pattern_sync_service
        self.hybrid_learning = pattern_sync_service.hybrid_learning
        self.task_type_service = pattern_sync_service.task_type_service
    
    async def complete_event_with_learning(self, 
                                         user_id: str,
                                         event_id: str,
                                         success_rating: float,
                                         energy_after: float,
                                         scheduled_start: datetime,
                                         task_type_id: str):
        """Complete an event and trigger learning"""
        
        # Record completion for behavioral learning
        await self.pattern_sync.record_task_completion(
            user_id=user_id,
            task_type_id=task_type_id,
            scheduled_start=scheduled_start,
            success_rating=success_rating,
            energy_after=energy_after
        )
        
        print(f"✅ Event completed and learning triggered for user {user_id}")
    
    async def get_scheduling_insights(self, user_id: str) -> Dict:
        """Get insights for better scheduling"""
        return await self.pattern_sync.get_learning_progress(user_id) 