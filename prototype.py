#!/usr/bin/env python3
"""
Smart Scheduler Prototype - Interactive Testing Script
Test the three-tier architecture with user input simulation
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

# Import our services
from task_type_service import TaskTypeService
from learning_service import LearningService
from scheduler_service import SchedulerService
from models import UserRole, ScheduleEventRequest, CompleteEventRequest

# Load environment variables
load_dotenv()

class SmartSchedulerPrototype:
    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([supabase_url, supabase_key, openai_api_key]):
            print("❌ Missing environment variables. Please check your .env file.")
            print("Required: SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY")
            exit(1)
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize services (Tier 2 + Learning)
        self.task_type_service = TaskTypeService(self.supabase, openai_api_key)
        self.learning_service = LearningService(self.task_type_service)
        self.scheduler_service = SchedulerService(self.task_type_service)
        
        # User context
        self.current_user_id = None
        self.scheduled_events = []  # In-memory event storage for prototype
        self.offline_mode = False  # Flag for offline/in-memory mode
        self.offline_task_types = {}  # In-memory task types storage
        
        print("🚀 Smart Scheduler Prototype Initialized!")
        print("📊 Three-tier architecture: Events → Task Types → Memory")
    
    async def setup_user(self):
        """Set up user profile for testing"""
        print("\n" + "="*60)
        print("🔧 USER SETUP")
        print("="*60)
        
        email = input("Enter your email: ").strip()
        
        print("\nSelect your role:")
        print("1. Student")
        print("2. Product Manager")
        print("3. Developer") 
        print("4. Executive")
        
        role_choice = input("Enter choice (1-4): ").strip()
        role_map = {
            "1": UserRole.STUDENT,
            "2": UserRole.PM,
            "3": UserRole.DEVELOPER,
            "4": UserRole.EXECUTIVE
        }
        
        role = role_map.get(role_choice, UserRole.DEVELOPER)
        
        # Create user in database
        try:
            # First try to find existing user by email
            existing_user = self.supabase.table("users").select("*").eq("email", email).execute()
            
            if existing_user.data:
                # User already exists, use existing user
                self.current_user_id = existing_user.data[0]["id"]
                print(f"✅ Found existing user: {email} ({existing_user.data[0]['role']})")
                print(f"📝 User ID: {self.current_user_id}")
            else:
                # Create new user
                result = self.supabase.table("users").insert({
                    "email": email,
                    "role": role.value,
                    "timezone": "UTC"
                }).execute()
                
                self.current_user_id = result.data[0]["id"]
                print(f"✅ User created: {email} ({role.value})")
                print(f"📝 User ID: {self.current_user_id}")
            
        except Exception as e:
            print(f"❌ Error with user setup: {e}")
            print("This might be due to database connection issues")
            print("Creating mock user for prototype testing...")
            
            # Generate a proper UUID for mock user
            self.current_user_id = str(uuid.uuid4())
            
            # Try to insert mock user into database for foreign key constraints
            try:
                mock_user_result = self.supabase.table("users").insert({
                    "id": self.current_user_id,
                    "email": f"mock-user-{self.current_user_id[:8]}@prototype.test",
                    "role": role.value,
                    "timezone": "UTC"
                }).execute()
                
                print(f"✅ Mock user created successfully")
                print(f"📝 Mock User ID: {self.current_user_id}")
                print("💡 You can now test all features including data persistence!")
                
            except Exception as mock_error:
                print(f"❌ Failed to create mock user: {mock_error}")
                print("💡 Foreign key constraints may cause issues, but learning algorithms can still be tested")
                print(f"📝 Mock User ID: {self.current_user_id}")
    
    async def show_current_patterns(self):
        """Display learned patterns for user's task types"""
        print("\n" + "="*60)
        print("📊 CURRENT LEARNED PATTERNS")
        print("="*60)
        
        task_types = await self.task_type_service.get_user_task_types(self.current_user_id)
        
        if not task_types:
            print("❌ No task types found. Schedule some tasks first!")
            return
        
        for task_type in task_types:
            print(f"\n🎯 Task Type: {task_type.task_type}")
            print(f"   Category: {task_type.category.value}")
            print(f"   Cognitive Load: {task_type.cognitive_load:.1f}")
            
            # Show learning progress
            analysis = self.learning_service.analyze_patterns(
                task_type.hourly_scores,
                task_type.confidence_scores
            )
            
            insights = self.learning_service.generate_pattern_insights(
                task_type.task_type, analysis
            )
            
            for insight in insights:
                print(f"   {insight}")
            
            # Show top 5 hours
            hourly_with_confidence = [
                (hour, score, conf) for hour, (score, conf) in 
                enumerate(zip(task_type.hourly_scores, task_type.confidence_scores))
            ]
            top_hours = sorted(hourly_with_confidence, key=lambda x: x[1], reverse=True)[:5]
            
            print(f"   🕐 Top 5 hours:")
            for hour, score, conf in top_hours:
                print(f"      {hour:02d}:00 - Score: {score:.2f} (confidence: {conf:.2f})")
    
    async def schedule_new_event(self):
        """Schedule a new event using the AI system"""
        print("\n" + "="*60)
        print("📅 SCHEDULE NEW EVENT")
        print("="*60)
        
        title = input("Event title: ").strip()
        description = input("Description (optional): ").strip() or None
        
        # Duration
        try:
            duration = float(input("Duration in hours (default: 1.0): ").strip() or "1.0")
        except ValueError:
            duration = 1.0
        
        # Preferred date
        preferred_input = input("Preferred date (YYYY-MM-DD) or press Enter for today: ").strip()
        preferred_date = None
        if preferred_input:
            try:
                preferred_date = datetime.strptime(preferred_input, "%Y-%m-%d")
            except ValueError:
                print("Invalid date format, using today")
        
        # Create scheduling request
        request = ScheduleEventRequest(
            title=title,
            description=description,
            duration=duration,
            preferred_date=preferred_date
        )
        
        print(f"\n🔍 Finding optimal time slot for '{title}'...")
        
        try:
            # Schedule the event
            result = await self.scheduler_service.schedule_event(self.current_user_id, request)
            
            # Display results
            event = result["event"]
            optimal_slot = result["optimal_slot"]
            task_type_info = result["task_type_used"]
            
            print(f"\n✅ EVENT SCHEDULED SUCCESSFULLY!")
            print(f"📅 Time: {event['scheduled_start'].strftime('%A, %B %d at %I:%M %p')}")
            print(f"⏱️  Duration: {duration} hours")
            print(f"🎯 Task Type: {task_type_info['name']} ({task_type_info['category']})")
            print(f"📊 Score: {optimal_slot['score']:.2f}")
            print(f"💡 Reasoning: {optimal_slot['reasoning']}")
            
            # Show pattern insights
            insights = result["pattern_insights"]
            if insights["best_hours"]:
                hours_str = ", ".join([f"{h}:00" for h in insights["best_hours"]])
                print(f"🎯 Generally best hours for this task: {hours_str}")
            
            # Store event for completion tracking
            event["completion_hour"] = event["scheduled_start"].hour
            self.scheduled_events.append(event)
            
            # Show alternatives
            if result["alternatives"]:
                print(f"\n🔄 Alternative time slots:")
                for i, alt in enumerate(result["alternatives"][:2], 1):
                    alt_time = alt['start_time'].strftime('%A, %B %d at %I:%M %p')
                    print(f"   {i}. {alt_time} (score: {alt['score']:.2f})")
            
        except Exception as e:
            print(f"❌ Error scheduling event: {e}")
    
    async def complete_event(self):
        """Mark an event as complete and provide feedback for learning"""
        print("\n" + "="*60)
        print("✅ COMPLETE EVENT")
        print("="*60)
        
        if not self.scheduled_events:
            print("❌ No events to complete. Schedule some events first!")
            return
        
        # Show scheduled events
        print("📋 Your scheduled events:")
        for i, event in enumerate(self.scheduled_events):
            if not event.get("completed", False):
                start_time = event["scheduled_start"].strftime('%m/%d %I:%M %p')
                print(f"{i+1}. {event['title']} - {start_time}")
        
        # Select event to complete
        try:
            choice = int(input("\nSelect event to complete (number): ")) - 1
            if choice < 0 or choice >= len(self.scheduled_events):
                print("❌ Invalid choice")
                return
            
            event = self.scheduled_events[choice]
            if event.get("completed", False):
                print("❌ Event already completed")
                return
            
        except ValueError:
            print("❌ Invalid input")
            return
        
        # Get completion feedback
        print(f"\n📝 Completing: {event['title']}")
        print("Please provide feedback (0.0 = worst, 1.0 = best):")
        
        try:
            success_rating = float(input("How well did it go? (0.0-1.0): "))
            energy_after = float(input("Energy level after completion? (0.0-1.0): "))
        except ValueError:
            print("❌ Invalid input, using defaults")
            success_rating = 0.7
            energy_after = 0.6
        
        # Update learning patterns
        print(f"\n🧠 Updating learned patterns...")
        
        await self.learning_service.update_task_type_patterns(
            event["task_type_id"],
            event["completion_hour"],
            success_rating > 0.7,
            energy_after,
            success_rating
        )
        
        # Mark event as completed
        event["completed"] = True
        event["success_rating"] = success_rating
        event["energy_after"] = energy_after
        
        print(f"✅ Event completed and patterns updated!")
    
    async def simulate_learning_data(self):
        """Simulate multiple completions to show learning in action"""
        print("\n" + "="*60)
        print("🧪 SIMULATE LEARNING DATA")
        print("="*60)
        
        print("This will simulate multiple task completions to demonstrate learning...")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm != 'y':
            return
        
        # Create a coding task type if it doesn't exist
        similar_task = await self.task_type_service.find_similar_task_type(
            self.current_user_id, "coding session programming development"
        )
        
        if not similar_task:
            from models import TaskCategory
            task_type = await self.task_type_service.create_task_type(
                self.current_user_id, 
                "Deep Coding Session", 
                TaskCategory.FOCUSED,
                "Programming and development work"
            )
        else:
            task_type = similar_task.task_type
        
        # Simulate completions at different hours
        simulations = [
            # Morning sessions (good)
            (8, True, 0.9),   # 8 AM - great success
            (9, True, 0.8),   # 9 AM - good success  
            (10, True, 0.7),  # 10 AM - decent
            
            # Afternoon sessions (mixed)
            (13, False, 0.4), # 1 PM - post-lunch crash
            (14, False, 0.3), # 2 PM - still low energy
            (15, True, 0.6),  # 3 PM - recovering
            
            # Evening sessions (okay)
            (19, True, 0.6),  # 7 PM - okay energy
            (20, False, 0.4), # 8 PM - getting tired
        ]
        
        print(f"📊 Simulating {len(simulations)} task completions...")
        
        for hour, success, energy in simulations:
            await self.learning_service.update_task_type_patterns(
                str(task_type.id),
                hour,
                success,
                energy
            )
        
        print("✅ Learning simulation complete!")
        print("🎯 Now you can see how the system learned your coding preferences")
    
    async def run_prototype(self):
        """Main prototype loop"""
        print("""
🎯 SMART SCHEDULER PROTOTYPE
Three-Tier Architecture Demo

📊 How it works:
• Tier 1: Events (what you schedule)
• Tier 2: Task Types (learned patterns) 
• Tier 3: Memory (context & insights)

🧠 The system learns from your completions and gets smarter over time!
        """)
        
        # Setup user
        await self.setup_user()
        
        while True:
            print("\n" + "="*60)
            print("🎛️  PROTOTYPE MENU")
            print("="*60)
            print("1. Schedule new event")
            print("2. Complete event (provides learning feedback)")
            print("3. View learned patterns")
            print("4. Simulate learning data (for demo)")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            try:
                if choice == "1":
                    await self.schedule_new_event()
                elif choice == "2":
                    await self.complete_event()
                elif choice == "3":
                    await self.show_current_patterns()
                elif choice == "4":
                    await self.simulate_learning_data()
                elif choice == "5":
                    print("👋 Thanks for testing the Smart Scheduler prototype!")
                    break
                else:
                    print("❌ Invalid choice, please try again")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or report this issue")


async def main():
    """Run the prototype"""
    prototype = SmartSchedulerPrototype()
    await prototype.run_prototype()


if __name__ == "__main__":
    print("🚀 Starting Smart Scheduler Prototype...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Prototype stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your environment setup and try again") 