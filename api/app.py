#!/usr/bin/env python3
"""
Smart Scheduler API Demo
Flask API that reuses existing scheduling services
"""

import os
import sys
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client

# Add parent directory to path to import our services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_type_service import TaskTypeService
from learning_service import LearningService
from scheduler_service import SchedulerService
from hybrid_learning_service import HybridLearningService
from models import UserRole, ScheduleEventRequest, CompleteEventRequest

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Default user credentials - always use these
DEFAULT_EMAIL = "x"
DEFAULT_ROLE = UserRole.STUDENT

class SmartSchedulerAPI:
    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([supabase_url, supabase_key, openai_api_key]):
            raise ValueError("Missing required environment variables")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_api_key = openai_api_key
        
        # Initialize services
        self.task_type_service = TaskTypeService(self.supabase, openai_api_key)
        self.learning_service = LearningService(self.task_type_service)
        self.scheduler_service = SchedulerService(self.task_type_service)
        
        # Initialize mem0 service if available
        try:
            from mem0_service import get_mem0_service
            self.mem0_service = get_mem0_service()
            if self.mem0_service and not self.mem0_service.is_available:
                asyncio.create_task(self.mem0_service.initialize_memory_service())
        except Exception as e:
            print(f"⚠️ Mem0 service not available: {e}")
            self.mem0_service = None
        
        # Auto-setup demo user with default credentials
        self.demo_user_id = self._setup_demo_user()
        print(f"🚀 API initialized with demo user: {self.demo_user_id}")
    
    def _setup_demo_user(self) -> str:
        """Automatically set up demo user with default email and role"""
        try:
            # Check if demo user already exists
            existing_user = self.supabase.table("users").select("*").eq("email", DEFAULT_EMAIL).execute()
            
            if existing_user.data:
                user_id = existing_user.data[0]["id"]
                print(f"✅ Using existing demo user: {DEFAULT_EMAIL}")
                return user_id
            else:
                # Create new demo user
                result = self.supabase.table("users").insert({
                    "email": DEFAULT_EMAIL,
                    "role": DEFAULT_ROLE.value,
                    "timezone": "UTC"
                }).execute()
                
                user_id = result.data[0]["id"]
                print(f"✅ Created demo user: {DEFAULT_EMAIL} ({DEFAULT_ROLE.value})")
                return user_id
                
        except Exception as e:
            print(f"⚠️ Error setting up demo user: {e}")
            # Generate a fallback UUID for demo purposes
            return str(uuid.uuid4())
    
    def _get_type_icon(self, event_type: str) -> str:
        """Get emoji icon for event type"""
        icon_map = {
            'meeting': '💼',
            'work': '💼',
            'study': '📚',
            'exercise': '🏃‍♂️',
            'health': '🏥',
            'personal': '👤',
            'social': '👥',
            'travel': '✈️',
            'food': '🍽️',
            'shopping': '🛒'
        }
        return icon_map.get(event_type.lower(), '📅')
    
    def _format_date_display(self, dt: datetime) -> str:
        """Format date for display: 'Wed, July 30'"""
        return dt.strftime("%a, %B %d")
    
    def _format_time_display(self, start_dt: datetime, end_dt: datetime) -> str:
        """Format time for display: '2:00pm - 2:30pm'"""
        # Use cross-platform format (Windows doesn't support %-I)
        start_time = start_dt.strftime("%I:%M%p").lower().lstrip('0')
        end_time = end_dt.strftime("%I:%M%p").lower().lstrip('0')
        return f"{start_time} - {end_time}"
    
    def _generate_priorities_array(self, current_priority: str) -> list:
        """Generate priorities array for chat event card display"""
        priorities = [
            {"label": "Low", "color": "low", "active": current_priority == "low"},
            {"label": "Med", "color": "medium", "active": current_priority == "medium"},
            {"label": "High", "color": "high", "active": current_priority == "high"}
        ]
        return priorities
    
    def _generate_availability_info(self) -> dict:
        """Generate availability information (mock data for demo)"""
        return {
            "label": "Available",
            "membersAvailable": "4/4 group members available",
            "conflicts": []  # Empty for demo
        }
    
    async def handle_chat_message(self, content: str, timestamp: str, user_id: str = None) -> dict:
        """Handle chat message and return BackendEventResponse"""
        try:
            # Always use default user regardless of provided user_id
            actual_user_id = self.demo_user_id
            
            # Classify the message
            classification = await self._classify_user_input(content)
            
            if classification['statement_type'] != 'schedule_event':
                return {
                    'success': True,
                    'message': 'I understand your message, but it doesn\'t seem to be a scheduling request. Try saying something like "Schedule a meeting" or "Book a workout session" to create an event!'
                }
            
            # Extract task details
            task_name = classification.get('extracted_task_name', content)
            
            # Create scheduling request
            request = ScheduleEventRequest(
                title=task_name,
                description=f"Event generated from: '{content}'",
                duration=1.0  # Default 1 hour
            )
            
            try:
                # Schedule the event
                result = await self.scheduler_service.schedule_event(
                    user_id=actual_user_id,
                    request=request
                )
                
                internal_event = result['event']
                
                # Parse dates for display formatting
                start_dt = internal_event['scheduled_start']
                end_dt = internal_event['scheduled_end']
                
                # Determine event type and priority
                event_type = classification.get('event_type', 'Meeting')
                priority_score = internal_event.get('calculated_priority', 0.5)
                
                if priority_score >= 0.7:
                    priority = 'high'
                elif priority_score >= 0.4:
                    priority = 'medium'
                else:
                    priority = 'low'
                
                # Determine category
                category = 'work'  # Default for meetings
                if any(word in content.lower() for word in ['exercise', 'workout', 'gym', 'health']):
                    category = 'health'
                elif any(word in content.lower() for word in ['personal', 'friend', 'family']):
                    category = 'personal'
                
                # Build BackendEventResponse
                event_response = {
                    'success': True,
                    'event': {
                        'id': internal_event.get('id', str(uuid.uuid4())),
                        'title': task_name,
                        'date': self._format_date_display(start_dt),
                        'time': self._format_time_display(start_dt, end_dt),
                        'startTime': start_dt.isoformat(),
                        'endTime': end_dt.isoformat(),
                        'type': event_type,
                        'typeIcon': self._get_type_icon(event_type),
                        'priority': priority,
                        'category': category,
                        'description': f"Meeting generated from: '{content}'",
                        'priorities': self._generate_priorities_array(priority),
                        'availability': self._generate_availability_info()
                    },
                    'message': "I've created a meeting proposal for you!"
                }
                
                return event_response
                
            except Exception as scheduling_error:
                print(f"⚠️ Scheduling failed: {scheduling_error}")
                print("🔄 Generating fallback event...")
                
                # Create fallback event when scheduling fails
                return self._create_fallback_event(task_name, content, classification)
            
        except Exception as e:
            print(f"❌ Chat message processing failed: {e}")
            return {
                'success': False,
                'message': f'Error processing chat message: {str(e)}'
            }
    
    def _create_fallback_event(self, task_name: str, content: str, classification: dict) -> dict:
        """Create a fallback event when scheduling fails"""
        # Generate simple future time slot
        start_time = datetime.now() + timedelta(hours=1)
        start_time = start_time.replace(minute=0, second=0, microsecond=0)  # Round to hour
        end_time = start_time + timedelta(hours=1)
        
        event_type = classification.get('event_type', 'Meeting')
        
        # Determine category
        category = 'work'  # Default for meetings
        if any(word in content.lower() for word in ['exercise', 'workout', 'gym', 'health']):
            category = 'health'
        elif any(word in content.lower() for word in ['personal', 'friend', 'family']):
            category = 'personal'
        
        return {
            'success': True,
            'event': {
                'id': str(uuid.uuid4()),
                'title': task_name,
                'date': self._format_date_display(start_time),
                'time': self._format_time_display(start_time, end_time),
                'startTime': start_time.isoformat(),
                'endTime': end_time.isoformat(),
                'type': event_type,
                'typeIcon': self._get_type_icon(event_type),
                'priority': 'medium',
                'category': category,
                'description': f"Event generated from: '{content}' (Fallback scheduling)",
                'priorities': self._generate_priorities_array('medium'),
                'availability': self._generate_availability_info()
            },
            'message': "I've created a meeting proposal for you! (Using fallback scheduling)"
        }
    
    async def handle_onboarding(self, preferences: list, timestamp: str, user_id: str = None) -> dict:
        """Handle onboarding preferences and return OnboardingResponse"""
        try:
            # Always use default user regardless of provided user_id
            actual_user_id = self.demo_user_id
            print(f"🎯 Using demo user ID: {actual_user_id}")
            
            # Process each preference like the prototype does
            task_types_created = 0
            preferences_applied = 0
            
            for preference in preferences:
                print(f"🔍 Processing preference: '{preference}'")
                
                # Classify the preference to extract task names (like prototype)
                classification = await self._classify_user_input(preference)
                task_name = classification.get('extracted_task_name', '').strip()
                extracted_preferences = classification.get('extracted_preferences', '').strip()
                
                print(f"📊 Classification results:")
                print(f"   📝 Task name: '{task_name}'")
                print(f"   🎯 Preferences: '{extracted_preferences}'")
                print(f"   🔍 Confidence: {classification.get('confidence', 0):.2f}")
                
                if task_name:
                    print(f"🎯 Processing task type for: '{task_name}'")
                    
                    try:
                        # Use exact same logic as prototype _handle_preference_from_onboarding
                        # 1. First check for exact task name match
                        existing_task_types = await self.task_type_service.get_user_task_types(actual_user_id)
                        exact_match = None
                        for existing_task in existing_task_types:
                            if existing_task.task_type.lower().strip() == task_name.lower().strip():
                                exact_match = existing_task
                                break
                        
                        if exact_match:
                            task_type = exact_match
                            print(f"🎯 ONBOARDING: Found exact task type match: {task_type.task_type}")
                        else:
                            # 2. Check for similar task type with RAG > 0.4
                            similar_task = await self.task_type_service.find_similar_task_type(
                                actual_user_id, task_name
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
                                        actual_user_id, task_name, 
                                        description=f"From onboarding: {preference}"
                                    )
                                    print(f"🆕 Created new task type: {task_type.task_type}")
                                    task_types_created += 1
                                except Exception as create_error:
                                    if "duplicate key" in str(create_error).lower():
                                        # Handle race condition like prototype
                                        print(f"⚠️ Task type '{task_name}' was created by another process, fetching existing...")
                                        existing_task_types = await self.task_type_service.get_user_task_types(actual_user_id)
                                        for existing_task in existing_task_types:
                                            if existing_task.task_type.lower().strip() == task_name.lower().strip():
                                                task_type = existing_task
                                                print(f"🔗 Using existing task type: {task_type.task_type}")
                                                break
                                        else:
                                            raise create_error
                                    else:
                                        raise create_error
                        
                        # Apply time preferences using the compact string format (like prototype)
                        if extracted_preferences:
                            print(f"🧠 Applying time preferences from classification...")
                            print(f"   🎯 Preferences pattern: '{extracted_preferences}'")
                            
                            # Parse the compact string format directly
                            parsed_patterns = self.scheduler_service._parse_time_pattern_string(extracted_preferences)
                            if parsed_patterns:
                                # Apply to behavioral arrays directly
                                updated = self.scheduler_service._apply_time_patterns_to_task_type(task_type, parsed_patterns)
                                if updated:
                                    # Save to database
                                    success = await self.scheduler_service._save_updated_task_type_patterns(task_type)
                                    if success:
                                        print(f"🎯 Applied {len(parsed_patterns)} time patterns to '{task_name}'!")
                                        for pattern in parsed_patterns:
                                            days_str = ",".join(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d] for d in pattern["days"])
                                            print(f"   {days_str} {pattern['hour_start']:02d}:00-{pattern['hour_end']:02d}:00 → boost {pattern['boost']:.1f}")
                                        preferences_applied += 1
                                    else:
                                        print(f"⚠️ Failed to save patterns to database")
                                else:
                                    print(f"⚠️ Could not apply time patterns")
                            else:
                                print(f"⚠️ Could not parse time pattern string: '{extracted_preferences}'")
                        else:
                            print(f"📝 No time preferences extracted from: '{preference}'")
                            
                    except Exception as task_error:
                        print(f"⚠️ Could not process task type for '{task_name}': {task_error}")
                else:
                    print(f"📝 No specific task name found in preference: '{preference}'")
                    if extracted_preferences:
                        print(f"   ⚠️ Time preferences found but no task name: '{extracted_preferences}'")
            
            # Store preferences in memory if available (like before)
            if self.mem0_service:
                print(f"🧠 Storing all preferences in Mem0...")
                preference_text = ", ".join(preferences)
                
                try:
                    await self.mem0_service.store_onboarding_preferences(
                        user_id=actual_user_id,
                        user_input=preference_text,
                        task_name="",
                        preferences=preferences,
                        openai_client=self.task_type_service.openai_client
                    )
                    print(f"✅ Mem0 storage successful for preferences")
                except Exception as mem0_error:
                    print(f"⚠️ Mem0 storage failed: {mem0_error}")
            else:
                print(f"⚠️ Mem0 service not available - preferences not stored in memory")
            
            print(f"✅ Onboarding completed successfully")
            print(f"   📊 Task types created: {task_types_created}")
            print(f"   🎯 Preferences applied: {preferences_applied}")
            
            return {
                'success': True,
                'message': f'Onboarding complete! Created {task_types_created} task types, applied {preferences_applied} preferences.',
                'userId': actual_user_id
            }
                
        except Exception as e:
            print(f"❌ Onboarding processing error: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing onboarding: {str(e)}',
                'userId': self.demo_user_id
            }
    
    async def _classify_user_input(self, user_input: str) -> dict:
        """Classify user input for scheduling with proper time preference extraction"""
        function_schema = {
            "name": "classify_scheduling_request",
            "description": "Classify user message for scheduling and extract time preferences",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement_type": {
                        "type": "string",
                        "enum": ["schedule_event", "preference_only"],
                        "description": "Whether user wants to schedule an event"
                    },
                    "extracted_task_name": {
                        "type": "string",
                        "description": "The event/meeting name to schedule"
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Type of event (Meeting, Call, Appointment, etc.)"
                    },
                    "extracted_preferences": {
                        "type": "string",
                        "description": "Time preferences in compact format: 'days:hour_start-hour_end:boost,days:hour_start-hour_end:boost'. Days: 0-6=Mon-Sun, 0-4=weekdays, 5-6=weekend. Hours: 0-23. Boost: 0.0-1.0. Example: '0-6:6-11:0.8,5-6:17-21:0.9'. Empty string if no preferences."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["statement_type", "extracted_task_name", "event_type", "extracted_preferences", "confidence"]
            }
        }
        
        context = f"""
Analyze this user statement and extract scheduling information:

User Statement: "{user_input}"

Classification Guidelines:
- "schedule_event": User wants to schedule something NOW (has task name + scheduling intent)
- "preference_only": User is stating preferences without immediate scheduling intent

Time Preference Format (CRITICAL - USE EXACT FORMAT):
Extract time preferences as compact string: "days:hour_start-hour_end:boost,days:hour_start-hour_end:boost"
- Days: 0-6 (0=Mon, 6=Sun), ranges like 0-4 (weekdays), 5-6 (weekend)
- Hours: 0-23 (24-hour format)
- Boost: 0.0-1.0 (0.0=avoid, 0.5=neutral, 1.0=prefer)

Examples:
- "I like morning workouts" → task: "workout", preferences: "0-6:6-11:0.8"
- "I prefer evening study sessions" → task: "study", preferences: "0-6:18-22:0.8"
- "I don't like Monday meetings" → task: "meeting", preferences: "0:0-23:0.2"
- "Weekend afternoon reading" → task: "reading", preferences: "5-6:12-17:0.8"
- "No specific time mentioned" → preferences: ""

Be precise with the format! Output ONLY the compact string for preferences.
"""

        try:
            response = self.task_type_service.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an intelligent scheduler. Extract task names and time preferences in the EXACT compact format specified."},
                    {"role": "user", "content": context}
                ],
                functions=[function_schema],
                function_call={"name": "classify_scheduling_request"},
                temperature=0.2
            )
            
            function_call = response.choices[0].message.function_call
            if function_call:
                import json
                return json.loads(function_call.arguments)
            else:
                raise ValueError("LLM did not return expected function call")
                
        except Exception as e:
            return {
                "statement_type": "schedule_event",
                "extracted_task_name": user_input,
                "event_type": "Meeting",
                "extracted_preferences": "",
                "confidence": 0.5
            }

# Initialize API instance
scheduler_api = SmartSchedulerAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'demo_user_id': scheduler_api.demo_user_id,
        'default_email': DEFAULT_EMAIL,
        'default_role': DEFAULT_ROLE.value
    })

@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    """Handle chat message for scheduling"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        user_id = data.get('userId')  # Will be ignored, always use default
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Missing content field'
            }), 400
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            scheduler_api.handle_chat_message(content, timestamp, user_id)
        )
        loop.close()
        
        # Always return 200 for valid requests, even if scheduling fails
        # Frontend will handle the success/failure based on result['success']
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/onboarding', methods=['POST'])
def onboarding():
    """Handle onboarding preferences"""
    try:
        data = request.get_json()
        preferences = data.get('preferences', [])
        timestamp = data.get('timestamp', datetime.now().isoformat())
        user_id = data.get('userId')  # Will be ignored, always use default
        
        print(f"👋 ONBOARDING REQUEST:")
        print(f"   📝 Preferences: {preferences}")
        print(f"   ⏰ Timestamp: {timestamp}")
        print(f"   👤 Provided UserId: {user_id} (will use default)")
        
        if not preferences:
            print(f"❌ ONBOARDING FAILED: Missing preferences field")
            return jsonify({
                'success': False,
                'message': 'Missing preferences field'
            }), 400
        
        print(f"🔄 Processing onboarding for {len(preferences)} preferences...")
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            scheduler_api.handle_onboarding(preferences, timestamp, user_id)
        )
        loop.close()
        
        if result['success']:
            print(f"✅ ONBOARDING SUCCESS: {result['message']}")
            print(f"   🎯 Demo User ID: {result['userId']}")
        else:
            print(f"⚠️ ONBOARDING PARTIAL FAILURE: {result['message']}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ONBOARDING SERVER ERROR: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

# Legacy endpoints for backward compatibility
@app.route('/onboarding', methods=['POST'])
def legacy_onboarding():
    """Legacy onboarding endpoint - redirects to new API"""
    return onboarding()

@app.route('/schedule', methods=['POST'])
def legacy_schedule():
    """Legacy schedule endpoint"""
    try:
        data = request.get_json()
        content = data.get('title', '') or data.get('description', '')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Missing title or description field'
            }), 400
        
        # Redirect to chat message handler
        return chat_message()
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/demo-status', methods=['GET'])
def demo_status():
    """Get current demo user status and learned patterns"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get user's task types
        task_types = loop.run_until_complete(
            scheduler_api.task_type_service.get_user_task_types(scheduler_api.demo_user_id)
        )
        
        loop.close()
        
        patterns = []
        for task_type in task_types:
            patterns.append({
                'task_type': task_type.task_type,
                'completion_count': task_type.completion_count,
                'importance_score': task_type.importance_score
            })
        
        return jsonify({
            'demo_user_id': scheduler_api.demo_user_id,
            'default_email': DEFAULT_EMAIL,
            'default_role': DEFAULT_ROLE.value,
            'learned_patterns': patterns,
            'total_task_types': len(task_types),
            'mem0_available': scheduler_api.mem0_service is not None
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting demo status: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Smart Scheduler API Demo...")
    print("📋 Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /api/chat/message - Process chat messages for scheduling")
    print("  POST /api/onboarding - Save user preferences")
    print("  GET  /demo-status - View learned patterns")
    print(f"🎯 Default User: {DEFAULT_EMAIL} ({DEFAULT_ROLE.value})")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000) 