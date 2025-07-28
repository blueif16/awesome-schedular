# Intelligent Scheduling Architecture

## Overview

A hybrid AI scheduling system that combines **behavioral pattern learning** with **semantic understanding** to provide optimal task scheduling. The system learns from user completions and natural language preferences to make intelligent scheduling decisions.

## Core Architecture

### 🧠 Dual Scheduling Strategy

The system operates with two distinct scheduling modes that automatically route based on context:

```
User Input → Similarity Check → Route Decision
    ↓
┌─────────────────┐    ┌─────────────────┐
│ schedule_with_  │    │ schedule_with_  │
│ habit()         │    │ llm()           │
│                 │    │                 │
│ Pure Behavioral │    │ Semantic +      │
│ Pattern Scoring │    │ Behavioral      │
└─────────────────┘    └─────────────────┘
```

### 🎯 Routing Logic

**Automatic Route Selection:**
- **High Similarity (>0.8)**: Use `schedule_with_habit()` - fast, deterministic
- **Low Similarity (<0.8)**: Use `schedule_with_llm()` - semantic understanding
- **User States Preferences**: Always use `schedule_with_llm()` - pattern extraction

## Detailed Flow

### 1. schedule_with_habit() - Behavioral Scheduling

**When Used:**
- Simple task requests: "schedule my workout"
- High task similarity (>0.8) found
- No explicit user preferences stated

**Process:**
```python
1. Query similar tasks via RAG/embedding search
2. IF similarity > 0.8:
   - Use existing task type behavioral patterns
   - Calculate slot scores from hourly_scores array
   - Apply cognitive load and recovery time factors
   - Return optimal slot with reasoning
3. ELSE:
   - Auto-fallback to schedule_with_llm()
```

**Data Sources:**
- `hourly_scores[24]` - Time preference patterns
- `confidence_scores[24]` - Data reliability
- `performance_by_hour[24]` - Energy level patterns
- `cognitive_load` - Task complexity
- `recovery_hours` - Time needed between tasks

### 2. schedule_with_llm() - Semantic Scheduling

**When Used:**
- New tasks (similarity <0.8)
- User provides preferences: "schedule meditation, but I'm not a morning person"
- Complex scheduling constraints

**Process:**
```python
1. Query Mem0 for existing user patterns
2. Extract available calendar slots
3. Prepare comprehensive LLM context:
   - User preferences (current request)
   - Historical patterns (from Mem0)
   - Behavioral data (task type patterns)
   - Calendar availability (free slots)
4. Call OpenAI with structured output
5. Parse LLM decision with reasoning
6. Extract and store new preferences in Mem0
```

**LLM Context Structure:**
```
TASK TO SCHEDULE: [title, duration, cognitive load]
USER PREFERENCES (CURRENT): [stated preferences]
HISTORICAL PATTERNS FROM MEMORY: [Mem0 context]
BEHAVIORAL PATTERNS: [task type data]
AVAILABLE FREE SLOTS: [calendar slots]
TODAY'S EXISTING EVENTS: [schedule context]
```

## Memory & Learning System

### 🧠 Mem0 Integration

**Query Phase (Retrieval):**
```python
search_queries = [
    f"scheduling preferences for {task_type}",
    f"time preferences for {category} tasks", 
    "scheduling patterns and preferences"
]
```

**Storage Phase (Learning):**
```python
# 1. Full user context
"When scheduling meditation: not a morning person"

# 2. Extracted patterns  
"User scheduling pattern: avoids mornings"

# 3. Specific time preferences
"User preference (morning_negative): not a morning person"
```

### 📊 Behavioral Pattern Learning

**Real-time Updates (No LLM):**
- Immediate habit score updates after task completion
- Adaptive learning rate (faster → slower as data matures)
- Regularization to prevent overfitting
- Weekly time slot confidence building

**Pattern Update Formula:**
```python
learning_rate = base_rate / (1 + completion_count * 0.1)
new_score = current_score * (1 - learning_rate) + success_signal * learning_rate

# Regularization for extreme values
if new_score > 0.8 or new_score < 0.2:
    new_score = new_score * 0.95 + 0.5 * 0.05
```

## Data Flow Examples

### Example 1: Onboarding Flow

**User Statement:** "I like to read books in the morning"

```
1. Input → schedule_with_llm (treating as scheduling request)
2. LLM extracts: task="read books", preference="morning"
3. Create task type with morning preference boost
4. Store in Mem0: "User prefers read books during morning"
5. Return: Scheduled optimal morning slot
```

**Later Request:** "read books"
```
1. Input → find_similar_task_type → similarity=0.95
2. Route → schedule_with_habit 
3. Use behavioral patterns (morning hours boosted)
4. Return: Optimal morning slot (no LLM needed)
```

### Example 2: Preference Learning

**User Statement:** "Schedule my meditation, but I'm not a morning person"

```
1. Input → schedule_with_llm (preference detected)
2. Query Mem0 → Previous patterns about meditation/mornings
3. LLM Context:
   - Current: "not a morning person"  
   - Historical: "User dislikes early activities"
   - Behavioral: Meditation task patterns
   - Calendar: Available evening slots
4. LLM Decision: "7PM Tuesday - aligns with evening preference"
5. Store Patterns:
   - "avoids mornings"
   - "prefers evening calm activities"
   - "User preference (morning_negative): not a morning person"
```

## Technical Components

### 🛠️ Core Services

**SchedulerService** (`scheduler_service.py`)
- Main scheduling logic and routing
- LLM integration with structured output
- Calendar slot extraction and analysis
- Pattern extraction and storage

**TaskTypeService** (`task_type_service.py`) 
- Task similarity matching via embeddings
- Task type creation and management
- Behavioral pattern storage

**HybridLearningService** (`hybrid_learning_service.py`)
- Completion-driven learning
- Pattern synchronization
- Real-time behavioral updates

### 🔧 Integration Points

**OpenAI Function Calling:**
```python
function_schema = {
    "name": "select_optimal_slot",
    "parameters": {
        "selected_slot_index": "int",
        "reasoning": "string", 
        "confidence": "float",
        "detected_patterns": "array"
    }
}
```

**Mem0 Storage Categories:**
- `scheduling_preference` - Full user statements
- `scheduling_pattern` - Extracted patterns
- `time_preference` - Specific time patterns
- `onboarding_preference` - Initial user preferences

## Key Advantages

### ⚡ Performance Benefits
- **Fast Scheduling**: Behavioral patterns enable sub-second scheduling for known tasks
- **Smart Fallbacks**: Automatic LLM routing when needed
- **Progressive Learning**: System becomes faster and more accurate over time

### 🎯 Intelligence Features  
- **Semantic Understanding**: "not a morning person" → avoid 6-11 AM slots
- **Context Retention**: Mem0 builds user preference profile over time
- **Adaptive Patterns**: Real completion data overrides initial assumptions
- **Confidence Scoring**: System knows when it's uncertain

### 🔄 Unified Workflow
- **Same Functions**: Onboarding and regular scheduling use identical flow
- **No Duplicate Logic**: Single codebase handles both behavioral and semantic
- **Consistent Storage**: All preferences flow through same Mem0 integration

## System Intelligence Levels

### Level 1: New User
- Default task type patterns
- LLM-driven scheduling for most requests
- High learning rate from completions

### Level 2: Learning User  
- Growing behavioral patterns
- Mixed LLM + behavioral scheduling
- Mem0 context building

### Level 3: Mature User
- Strong behavioral patterns (>15 completions per task)
- Mostly habit-driven scheduling
- LLM only for new tasks or explicit preferences
- Rich Mem0 context for edge cases

## Future Enhancements

### Planned Features
- **Cross-task pattern learning**: "Morning workouts affect afternoon focus"
- **Seasonal adaptations**: "User prefers outdoor tasks in summer"
- **Energy optimization**: "Schedule demanding tasks during peak energy"
- **Context-aware scheduling**: "Avoid calls during commute hours"

### Technical Roadmap
- Enhanced confidence scoring in behavioral patterns
- Multi-user pattern sharing (privacy-aware)
- Integration with external calendars
- Mobile app with quick preference capture

---

*This architecture enables intelligent, personalized scheduling that learns from both explicit user preferences and implicit behavioral patterns, providing optimal task placement with minimal user intervention.* 