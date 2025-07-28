#!/usr/bin/env python3
"""
Smart Scheduler Prototype - Interactive Testing Script
Test the three-tier architecture with user input simulation
"""

import asyncio
import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict
from dotenv import load_dotenv
from supabase import create_client, Client

# Import our services
from task_type_service import TaskTypeService
from learning_service import LearningService
from scheduler_service import SchedulerService
from hybrid_learning_service import HybridLearningService
from models import UserRole, ScheduleEventRequest, CompleteEventRequest

# Load environment variables
load_dotenv()

class SmartSchedulerPrototype:
    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        # Use SERVICE_ROLE key for prototyping to bypass RLS
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([supabase_url, supabase_key, openai_api_key]):
            print("❌ Missing environment variables. Please check your .env file.")
            print("Required: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY")
            print("💡 Make sure you're using SERVICE_ROLE key, not ANON key for prototyping")
            exit(1)
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_api_key = openai_api_key  # Store for onboarding functionality
        
        # Initialize services (Tier 2 + Learning)
        self.task_type_service = TaskTypeService(self.supabase, openai_api_key)
        self.learning_service = LearningService(self.task_type_service)
        self.scheduler_service = SchedulerService(self.task_type_service)
        
        # Initialize hybrid learning service (completion-based mem0 updates)
        # Note: For prototype, we'll create a simple memory service mock
        try:
            from mem0_service import get_mem0_service
            self.mem0_service = get_mem0_service()
            self.hybrid_learning_service = None  # Initialize later
        except Exception as e:
            print(f"⚠️ Could not initialize Mem0 service: {e}")
            print("   Prototype will work without mem0 updates")
            self.hybrid_learning_service = None
            self.mem0_service = None
        
        # User context
        self.current_user_id = None
        self.scheduled_events = []  # In-memory event storage for prototype
        self.offline_mode = False  # Flag for offline/in-memory mode
        self.offline_task_types = {}  # In-memory task types storage
        
        print("🚀 Smart Scheduler Prototype Initialized!")
        print("📊 Three-tier architecture: Events → Task Types → Memory")
    
    def _get_next_clean_hour(self, dt: datetime) -> datetime:
        """Get next clean hour boundary from given datetime"""
        if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0:
            return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            return dt.replace(minute=0, second=0, microsecond=0)

    async def setup_user(self):
        """Set up user profile for testing"""
        print("\n" + "="*60)
        print("🔧 USER SETUP")
        print("="*60)
        
        # Initialize mem0 service if available
        if self.mem0_service and not self.mem0_service.is_available:
            await self.mem0_service.initialize_memory_service()
            
        # Set task_type_service for database persistence
        if self.mem0_service:
            from mem0_service import set_task_type_service
            set_task_type_service(self.task_type_service)
            
        # Initialize hybrid learning service if mem0 is available
        if self.mem0_service and self.mem0_service.is_available:
            try:
                self.hybrid_learning_service = HybridLearningService(
                    self.task_type_service, 
                    self.mem0_service.memory_service, 
                    self.openai_api_key
                )
                print("✅ Hybrid learning service initialized with mem0")
            except Exception as e:
                print(f"⚠️ Could not initialize hybrid learning service: {e}")
        
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
            if task_type.description:
                print(f"   Description: {task_type.description}")
            print(f"   Total Completions: {task_type.completion_count}")
            print(f"   Since Last Update: {task_type.completions_since_last_update}")
            print(f"   Importance Score: {task_type.importance_score:.2f}")
            print(f"   Recovery Hours: {task_type.recovery_hours:.1f}")
            
            # Show highest confidence time slots
            if task_type.slot_confidence:
                max_confidence = 0
                best_slots = []
                for day in range(7):
                    for hour in range(24):
                        confidence = task_type.slot_confidence[day][hour]
                        if confidence > 0.5:  # Only show slots with decent confidence
                            day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                            best_slots.append((f"{day_names[day]} {hour:02d}:00", confidence))
                
                if best_slots:
                    best_slots.sort(key=lambda x: x[1], reverse=True)
                    print(f"   🎯 High Confidence Slots:")
                    for slot, confidence in best_slots[:5]:
                        print(f"      {slot}: {confidence:.2f}")
            
            # Show learning progress using new weekly analysis
            analysis = self.learning_service.analyze_weekly_patterns(
                task_type.weekly_habit_scores,
                task_type.completion_count
            )
            
            insights = self.learning_service.generate_weekly_pattern_insights(
                task_type.task_type, analysis
            )
            
            for insight in insights:
                print(f"   {insight}")
            
            # Show if ready for mem0 update
            if task_type.completions_since_last_update >= 5:
                print(f"   🧠 Ready for Mem0 update (5+ completions)")
            elif task_type.completions_since_last_update > 0:
                remaining = 5 - task_type.completions_since_last_update
                print(f"   📊 {remaining} more completions until Mem0 update")
    
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
        
        # Choose scheduling mode
        print("\nScheduling mode:")
        print("1. Next 7 days (default)")
        print("2. Today only")
        print("3. Single deadline")
        print("4. Custom time periods")
        
        mode_choice = input("Enter choice (1-4, default: 1): ").strip() or "1"
        
        available_periods = None
        preferred_date = None
        
        if mode_choice == "1":
            # Default 7-day window starting tomorrow
            preferred_input = input("Preferred start date (YYYY-MM-DD) or press Enter for tomorrow: ").strip()
            if preferred_input:
                try:
                    preferred_date = datetime.strptime(preferred_input, "%Y-%m-%d")
                except ValueError:
                    print("Invalid date format, using tomorrow")
                    preferred_date = datetime.now() + timedelta(days=1)
            else:
                # Default to tomorrow when no input provided
                preferred_date = datetime.now() + timedelta(days=1)
        
        elif mode_choice == "2":
            # Today only - start from next clean hour
            now = datetime.now()
            start_time = self._get_next_clean_hour(now)
            end_of_day = now.replace(hour=23, minute=59, second=59)
            available_periods = [(start_time, end_of_day)]
            print(f"📅 Scheduling for today only: {start_time.strftime('%m/%d %H:%M')} - {end_of_day.strftime('%H:%M')}")
        
        elif mode_choice == "3":
            # Single deadline
            deadline_input = input("Deadline (YYYY-MM-DD HH:MM): ").strip()
            try:
                deadline = datetime.strptime(deadline_input, "%Y-%m-%d %H:%M")
                now = datetime.now()
                if deadline > now:
                    # Start from next clean hour
                    start_time = self._get_next_clean_hour(now)
                    available_periods = [(start_time, deadline)]
                    print(f"📅 Scheduling before deadline: {start_time.strftime('%m/%d %H:%M')} - {deadline.strftime('%m/%d %H:%M')}")
                else:
                    print("❌ Deadline must be in the future")
                    return
            except ValueError:
                print("❌ Invalid deadline format (YYYY-MM-DD HH:MM)")
                return
        
        elif mode_choice == "4":
            # Custom time periods
            print("Enter available time periods (press Enter when done):")
            periods = []
            period_num = 1
            
            while True:
                period_input = input(f"Period {period_num} (YYYY-MM-DD HH:MM to YYYY-MM-DD HH:MM): ").strip()
                if not period_input:
                    break
                
                try:
                    start_str, end_str = period_input.split(" to ")
                    start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
                    end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
                    
                    if end_time > start_time:
                        periods.append((start_time, end_time))
                        print(f"  ✅ Added period: {start_time.strftime('%m/%d %H:%M')} - {end_time.strftime('%m/%d %H:%M')}")
                        period_num += 1
                    else:
                        print("  ❌ End time must be after start time")
                except ValueError:
                    print("  ❌ Invalid format. Use: YYYY-MM-DD HH:MM to YYYY-MM-DD HH:MM")
            
            if periods:
                available_periods = periods
            else:
                print("No valid periods entered, using default 7-day window")
        
        # Create scheduling request
        request = ScheduleEventRequest(
            title=title,
            description=description,
            duration=duration,
            preferred_date=preferred_date
        )
        
        print(f"\n🔍 Finding optimal time slot for '{title}'...")
        
        try:
            # Schedule the event with available periods
            result = await self.scheduler_service.schedule_event(
                self.current_user_id, 
                request,
                available_periods
            )
            
            # Display results
            event = result["event"]
            optimal_slot = result["optimal_slot"]
            task_type_info = result["task_type_used"]
            stats = result.get("scheduling_stats", {})
            
            print(f"\n✅ EVENT SCHEDULED SUCCESSFULLY!")
            print(f"📅 Time: {event['scheduled_start'].strftime('%A, %B %d at %I:%M %p')}")
            print(f"⏱️  Duration: {duration} hours")
            print(f"🎯 Task Type: {task_type_info['name']}")
            print(f"📊 Score: {optimal_slot['score']:.2f}")
            print(f"💡 Reasoning: {optimal_slot['reasoning']}")
            
            # Show scheduling stats
            if stats.get("periods_searched"):
                print(f"🔍 Searched {stats['periods_searched']} time periods, found {stats['total_candidates']} candidates")
            
            # Show which period was selected
            if "period" in optimal_slot:
                print(f"📍 Selected from period: {optimal_slot['period']}")
            
            # Show pattern insights
            insights = result["pattern_insights"]
            if insights.get("best_hours"):
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
                    alt_period = alt.get('period', 'N/A')
                    print(f"   {i}. {alt_time} (score: {alt['score']:.2f}) in {alt_period}")
            
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
            success_rating,
            self.current_user_id,  # Pass user_id for hybrid learning
            self.hybrid_learning_service  # Pass hybrid service for completion tracking
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
        
        # Use simplified logic with new schema
        task_name = "Deep Coding Session"
        task_description = "Programming and development work requiring focused attention"
        
        # 1. Find similar task type
        similar_task = await self.task_type_service.find_similar_task_type(
            self.current_user_id, task_name, task_description
        )
        
        if similar_task and similar_task.similarity > 0.4:
            print(f"🎯 SIMULATION: Using existing task type '{similar_task.task_type.task_type}' (completions: {similar_task.task_type.completion_count})")
            task_type = similar_task.task_type
        else:
            # 2. Create new task type with description
            if similar_task:
                print(f"🔄 Similarity {similar_task.similarity:.3f} < 0.4 threshold - creating new task type for simulation")
            print(f"🆕 Creating new task type for simulation: '{task_name}'")
            try:
                task_type = await self.task_type_service.create_task_type(
                    self.current_user_id, task_name, task_description
                )
            except Exception as create_error:
                if "duplicate key" in str(create_error).lower():
                    # Handle race condition - task was created by another process
                    print(f"⚠️ Task type '{task_name}' already exists, fetching existing...")
                    similar_task_retry = await self.task_type_service.find_similar_task_type(
                        self.current_user_id, task_name, task_description
                    )
                    if similar_task_retry and similar_task_retry.similarity > 0.7:
                        task_type = similar_task_retry.task_type
                        print(f"🔗 RACE CONDITION RESOLVED: Using existing task type '{task_type.task_type}' (similarity: {similar_task_retry.similarity:.3f})")
                    else:
                        print(f"❌ Race condition but similarity too low: {similar_task_retry.similarity:.3f} < 0.7")
                        raise create_error  # Re-raise if we can't find a good match
                else:
                    raise create_error
        
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
    
    async def classify_user_input(self, user_input: str) -> Dict:
        """Classify user input to determine if it's a scheduling request or preference statement"""
        function_schema = {
            "name": "classify_user_statement",
            "description": "Classify user statement as scheduling request or preference statement",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement_type": {
                        "type": "string",
                        "enum": ["schedule_event", "preference_only"],
                        "description": "Whether user wants to schedule an event or just stating preferences"
                    },
                    "has_task_name": {
                        "type": "boolean", 
                        "description": "Whether the statement contains a specific task name or activity"
                    },
                    "has_schedule_intent": {
                        "type": "boolean",
                        "description": "Whether user explicitly wants to schedule something now"
                    },
                    "extracted_task_name": {
                        "type": "string",
                        "description": "The task/activity name if present, empty string if not found"
                    },
                    "extracted_duration": {
                        "type": "number",
                        "description": "Duration in hours if mentioned, 1.0 as default if not specified"
                    },
                    "extracted_preferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of time/scheduling preferences detected (e.g., 'morning', 'evening', 'after coffee')"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score in classification (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["statement_type", "has_task_name", "has_schedule_intent", "extracted_task_name", "extracted_duration", "extracted_preferences", "confidence"]
            }
        }
        
        context = f"""
Analyze this user statement and classify it:

User Statement: "{user_input}"

Classification Guidelines:
- "schedule_event": User wants to schedule something NOW (has task name + scheduling intent)
  Examples: "Schedule coding for 2 hours", "I want to read for 1 hour tomorrow"
  
- "preference_only": User is stating preferences without immediate scheduling intent
  Examples: "I like reading in the morning", "I prefer coding in evenings", "I'm not a morning person"

Extract:
- Task name: specific activity mentioned
- Duration: time mentioned or default 1.0 hours  
- Preferences: time-related preferences like "morning", "evening", "after coffee"
- Schedule intent: explicit words like "schedule", "plan", "book", "want to do"
"""
        
        try:
            response = self.task_type_service.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an intelligent scheduler that classifies user statements."},
                    {"role": "user", "content": context}
                ],
                functions=[function_schema],
                function_call={"name": "classify_user_statement"},
                temperature=0.2
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "classify_user_statement":
                return json.loads(function_call.arguments)
            else:
                raise ValueError("LLM did not return expected function call")
                
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            # Fallback classification
            return {
                "statement_type": "preference_only",
                "has_task_name": False,
                "has_schedule_intent": False,
                "extracted_task_name": "",
                "extracted_duration": 1.0,
                "extracted_preferences": [],
                "confidence": 0.3
            }
    
    async def onboarding(self):
        """Onboarding flow to classify user statement and decide action"""
        print("\n" + "="*60)
        print("👋 ONBOARDING")
        print("="*60)
        print("""
🎯 Smart Onboarding System

Tell me about your preferences or what you'd like to schedule!

Examples:
• "I like to read books in the morning" (preference)
• "Schedule meditation for 30 minutes" (event)
• "I want to code for 2 hours tomorrow" (event)
• "I'm not a morning person for meetings" (preference)

Type 'exit' to return to main menu.
        """)
        
        while True:
            user_input = input("\n💬 Your statement: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'back']:
                print("👋 Returning to main menu.")
                break
                
            if not user_input:
                print("❌ Please enter a statement.")
                continue
            
            print(f"\n🔍 Analyzing: '{user_input}'")
            
            # Classify the user input
            classification = await self.classify_user_input(user_input)
            
            print(f"📊 Classification: {classification['statement_type']} (confidence: {classification['confidence']:.2f})")
            
            if classification['statement_type'] == 'schedule_event':
                # User wants to create an event
                await self._handle_schedule_from_onboarding(user_input, classification)
            else:
                # User is stating preferences only
                await self._handle_preference_from_onboarding(user_input, classification)
    
    async def _handle_schedule_from_onboarding(self, user_input: str, classification: Dict):
        """Handle scheduling request from onboarding"""
        print(f"\n✅ SCHEDULING EVENT FROM ONBOARDING")
        
        task_name = classification['extracted_task_name']
        duration = classification['extracted_duration']
        preferences = " ".join(classification['extracted_preferences'])
        
        if not task_name:
            print("❌ Could not extract task name. Please be more specific.")
            return
        
        print(f"📝 Task: {task_name}")
        print(f"⏱️  Duration: {duration} hours")
        if preferences:
            print(f"🎯 Preferences: {preferences}")
        
        # Create scheduling request
        request = ScheduleEventRequest(
            title=task_name,
            description=f"Created from onboarding: {user_input}",
            duration=duration
        )
        
        try:
            # Use LLM scheduling with the user preferences
            result = await self.scheduler_service.schedule_with_llm(
                user_id=self.current_user_id,
                request=request,
                user_preferences=user_input,
                existing_events=self.scheduled_events,
                openai_client=self.task_type_service.openai_client,
                memory_service=self.mem0_service.memory_service if self.mem0_service else None
            )
            
            # Display results
            event = result["event"]
            print(f"\n✅ EVENT SCHEDULED!")
            print(f"📅 Time: {event['scheduled_start'].strftime('%A, %B %d at %I:%M %p')}")
            print(f"💡 LLM Reasoning: {result.get('llm_reasoning', 'No reasoning provided')}")
            
            if result.get('detected_patterns'):
                print(f"🎯 Detected Patterns: {', '.join(result['detected_patterns'])}")
            
            # Store event for completion tracking
            self.scheduled_events.append(event)
            
            print(f"🧠 Preferences stored in memory for future scheduling!")
            
            # Also apply preferences to task type behavioral patterns if detected
            if classification['extracted_preferences']:
                try:
                    # Get the task type that was used/created for this event
                    task_type = await self.task_type_service.get_task_type(event["task_type_id"])
                    if task_type:
                        print(f"🔄 Also updating behavioral patterns for faster future scheduling...")
                        
                        success = await self.scheduler_service.parse_and_apply_time_preferences(
                            user_id=self.current_user_id,
                            task_type=task_type,
                            user_preferences=user_input,
                            openai_client=self.task_type_service.openai_client
                        )
                        
                        if success:
                            print(f"✅ Future '{task_name}' scheduling will be even smarter!")
                        else:
                            print(f"⚠️ Could not extract behavioral patterns from preferences")
                except Exception as pattern_error:
                    print(f"⚠️ Could not update behavioral patterns: {pattern_error}")
            
        except Exception as e:
            print(f"❌ Error scheduling event: {e}")
    
    async def _handle_preference_from_onboarding(self, user_input: str, classification: Dict):
        """Handle preference statement from onboarding"""
        print(f"\n📝 STORING PREFERENCES")
        
        preferences = classification['extracted_preferences']
        task_name = classification.get('extracted_task_name', '')
        
        if preferences:
            print(f"🎯 Detected preferences: {', '.join(preferences)}")
        
        if task_name:
            print(f"📋 For task type: {task_name}")
        
        # Store in memory if available
        if self.mem0_service:
            try:
                await self.mem0_service.store_onboarding_preferences(
                    user_id=self.current_user_id,
                    user_input=user_input,
                    task_name=task_name,
                    preferences=preferences,
                    openai_client=self.task_type_service.openai_client
                )
                
                print(f"✅ Stored {len(preferences or []) + 1} preference items in memory!")
                print(f"🎯 These will help optimize future scheduling for you.")
                
            except Exception as e:
                print(f"⚠️ Could not store in memory: {e}")
                print(f"📝 Preferences noted locally: {user_input}")
        else:
            print(f"📝 Preferences noted: {user_input}")
            print(f"⚠️ Memory service not available - stored locally only")
        
        # If task name detected, create or update task type
        if task_name:
            try:
                # 1. First check for exact task name match
                existing_task_types = await self.task_type_service.get_user_task_types(self.current_user_id)
                exact_match = None
                for existing_task in existing_task_types:
                    if existing_task.task_type.lower().strip() == task_name.lower().strip():
                        exact_match = existing_task
                        break
                
                if exact_match:
                    task_type = exact_match
                    print(f"🎯 ONBOARDING: Found exact task type match: {task_type.task_type}")
                else:
                    # 2. Check for similar task type with RAG > 0.8
                    similar_task = await self.task_type_service.find_similar_task_type(
                        self.current_user_id, task_name
                    )
                    
                    if similar_task and similar_task.similarity > 0.4:
                        task_type = similar_task.task_type
                        print(f"🎯 ONBOARDING: Using similar task type '{task_type.task_type}' (completions: {task_type.completion_count})")
                    else:
                        # 3. Only create new if no exact or similar match found
                        if similar_task:
                            print(f"🔄 Similarity {similar_task.similarity:.3f} < 0.4 threshold - creating new task type")
                        print(f"🆕 Creating new task type for onboarding: '{task_name}'")
                        try:
                            task_type = await self.task_type_service.create_task_type(
                                self.current_user_id, task_name, 
                                description=f"{user_input}"
                            )
                            print(f"🆕 Created new task type: {task_type.task_type}")
                        except Exception as create_error:
                            if "duplicate key" in str(create_error).lower():
                                # Handle race condition - task was created by another process
                                print(f"⚠️ Task type '{task_name}' was created by another process, fetching existing...")
                                existing_task_types = await self.task_type_service.get_user_task_types(self.current_user_id)
                                for existing_task in existing_task_types:
                                    if existing_task.task_type.lower().strip() == task_name.lower().strip():
                                        task_type = existing_task
                                        print(f"🔗 Using existing task type: {task_type.task_type}")
                                        break
                                else:
                                    raise create_error  # Re-raise if we still can't find it
                            else:
                                raise create_error
                
                # Apply time preferences using LLM-powered parsing
                if preferences:
                    print(f"🧠 Analyzing time preferences with AI...")
                    preferences_text = " ".join(preferences) + " " + user_input
                    
                    success = await self.scheduler_service.parse_and_apply_time_preferences(
                        user_id=self.current_user_id,
                        task_type=task_type,
                        user_preferences=preferences_text,
                        openai_client=self.task_type_service.openai_client
                    )
                    
                    if success:
                        print(f"🎯 Future scheduling for '{task_name}' will use these learned preferences!")
                    else:
                        print(f"⚠️ Could not extract specific time preferences from input")
                else:
                    print(f"📝 Task type created without specific time preferences")
                
            except Exception as e:
                print(f"⚠️ Could not create task type: {e}")
    
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
            print("5. Onboarding (classify user input)")
            print("6. Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
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
                    await self.onboarding()
                elif choice == "6":
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