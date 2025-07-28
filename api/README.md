# Smart Scheduler API

Flask API that provides intelligent scheduling capabilities using AI-powered pattern learning and preference analysis.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp ../env_template.txt .env
# Edit .env with your actual values

# Run the API
python app.py
```

The API will start on `http://localhost:5000` with the following endpoints.

## 📋 API Endpoints

### Health Check
```http
GET /health
```

Returns API status and default user information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-XX...",
  "demo_user_id": "uuid",
  "default_email": "x",
  "default_role": "student"
}
```

### Chat Message Scheduling
```http
POST /api/chat/message
```

Process chat messages for event scheduling.

**Request:**
```json
{
  "content": "Schedule a meeting with the team",
  "timestamp": "2025-01-XX...",
  "userId": "user-123"  // Optional, always uses default user
}
```

**Response (BackendEventResponse):**
```json
{
  "success": true,
  "event": {
    "id": "event-123456",
    "title": "Team Meeting",
    "date": "Wed, July 30",
    "time": "2:00pm - 2:30pm",
    "startTime": "2025-07-30T14:00:00.000Z",
    "endTime": "2025-07-30T14:30:00.000Z",
    "type": "Meeting",
    "typeIcon": "💼",
    "priority": "medium",
    "category": "work",
    "description": "Meeting generated from: 'Schedule a meeting with the team'",
    "priorities": [
      {"label": "Low", "active": false, "color": "low"},
      {"label": "Med", "active": true, "color": "medium"},
      {"label": "High", "active": false, "color": "high"}
    ],
    "availability": {
      "label": "Available",
      "membersAvailable": "4/4 group members available",
      "conflicts": []
    }
  },
  "message": "I've created a meeting proposal for you!"
}
```

### Onboarding Preferences
```http
POST /api/onboarding
```

Save user preferences during onboarding.

**Request:**
```json
{
  "preferences": ["Morning exercise", "Learn Spanish"],
  "timestamp": "2025-01-XX...",
  "userId": "user-123"  // Optional, always uses default user
}
```

**Response (OnboardingResponse):**
```json
{
  "success": true,
  "message": "Onboarding preferences saved successfully!",
  "userId": "user-123456"
}
```

### Demo Status
```http
GET /demo-status
```

Get current user patterns and learning status.

**Response:**
```json
{
  "demo_user_id": "uuid",
  "default_email": "x",
  "default_role": "student",
  "learned_patterns": [
    {
      "task_type": "meeting",
      "completion_count": 5,
      "importance_score": 0.8
    }
  ],
  "total_task_types": 3,
  "mem0_available": true
}
```

## 🎯 Default User System

The API always uses consistent default credentials regardless of any `userId` provided in requests:

- **Email:** `x`
- **Role:** `student`
- **User ID:** Auto-generated UUID

This ensures consistent behavior across all API calls and simplifies testing.

## 📝 Event Types & Categories

### Event Types
- **Meeting** 💼 - Team meetings, sync calls
- **Study** 📚 - Learning sessions, courses
- **Exercise** 🏃‍♂️ - Workouts, gym sessions
- **Health** 🏥 - Doctor appointments, checkups
- **Personal** 👤 - Personal tasks, calls

### Categories
- **work** - Professional activities
- **health** - Fitness and medical
- **personal** - Personal activities

### Priorities
- **high** - Priority score ≥ 0.7
- **medium** - Priority score 0.4-0.7
- **low** - Priority score < 0.4

## 🧪 Testing

Run the test suite to verify all endpoints:

```bash
python test_api.py
```

Example test outputs:
```bash
🧪 Smart Scheduler API Test Suite
==================================================
🩺 Testing health check...
Status: 200

💬 Testing chat message endpoint...
✅ Created: Team Meeting - Wed, January 29 2:00pm - 3:00pm
   Priority: medium, Category: work

👋 Testing onboarding endpoint...
✅ Preferences saved successfully!
```

## 🔧 Legacy Endpoints

Backward compatibility endpoints are maintained:
- `POST /onboarding` → redirects to `/api/onboarding`
- `POST /schedule` → redirects to `/api/chat/message`

## 🏗️ Architecture

The API integrates with several backend services:

- **TaskTypeService** - Event classification and learning
- **LearningService** - Pattern recognition from user behavior
- **SchedulerService** - Optimal time slot calculation
- **Mem0Service** - Long-term memory and preferences (optional)

## 🌐 CORS Support

CORS is enabled for frontend integration. The API accepts requests from any origin during development.

## 🚨 Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "message": "Error description"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad Request (missing required fields)
- `500` - Internal Server Error 