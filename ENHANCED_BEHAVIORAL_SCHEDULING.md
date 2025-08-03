# Enhanced Behavioral Scheduling System

## 🎯 **Overview**

The Enhanced Behavioral Scheduling System represents a significant evolution from simple habit-based scheduling to a comprehensive behavioral pattern analysis. This system considers **two key behavioral factors** to make optimal scheduling decisions:

1. **Task Habit Patterns** (50%) - When users typically perform specific tasks
2. **Energy-Cognitive Fit** (50%) - User energy patterns adjusted for task cognitive load requirements

---

## 🔄 **System Evolution**

### **Before: `schedule_with_habit()`**
- ✅ Task habit patterns only
- ✅ Basic time preference scoring
- ❌ No energy consideration
- ❌ No cognitive load matching
- ❌ No duration optimization

### **After: `schedule_with_behavioral_patterns()`**
- ✅ **Two-factor behavioral analysis (habit + energy-cognitive fit)**
- ✅ **Smart energy-cognitive load matching** 
- ✅ **Enhanced reasoning generation**
- ✅ **Comprehensive scoring insights**

---

## 📊 **Scoring Algorithm**

### **Core Formula**
```python
base_score = (habit_preference * 0.5) + (energy_cognitive_fit * 0.5)
final_score = base_score * confidence * recovery_penalty
```

### **Factor Breakdown**

#### **1. Habit Component (50% weight)**
```python
habit_component = weekly_habit_scores[weekly_index] * 0.5
# weekly_index = day_of_week * 24 + hour (0-167)
```

#### **2. Energy-Cognitive Fit Component (50% weight)**  
```python
# Adjust energy score based on task's cognitive demands
if cognitive_load > 0.7:  # High cognitive load task
    # High-cognitive tasks need high energy - penalize low energy periods
    energy_cognitive_fit = energy_score  # Direct energy score (0.0-1.0)
elif cognitive_load < 0.3:  # Low cognitive load task  
    # Low-cognitive tasks can work with lower energy - boost low energy periods
    energy_cognitive_fit = 1.0 - (1.0 - energy_score) * 0.5  # Reduce penalty for low energy
else:  # Medium cognitive load
    # Medium-cognitive tasks have moderate energy requirements
    energy_cognitive_fit = energy_score * 0.8 + 0.2  # Slight boost across all energy levels

energy_component = energy_cognitive_fit * 0.5
```

---

## 🧠 **Energy-Cognitive Fit Examples**

### **High-Cognitive Task (0.9) + High Energy (0.9)**
```
energy_cognitive_fit = 0.9  # Direct energy score - needs high energy
energy_component = 0.9 * 0.5 = 0.45
→ "High-cognitive task during high energy (excellent fit)"
```

### **High-Cognitive Task (0.9) + Low Energy (0.3)**
```
energy_cognitive_fit = 0.3  # Direct energy score - poor for high-cognitive tasks
energy_component = 0.3 * 0.5 = 0.15
→ "High-cognitive task during low energy (poor fit)"
```

### **Low-Cognitive Task (0.2) + Low Energy (0.3)**
```
energy_cognitive_fit = 1.0 - (1.0 - 0.3) * 0.5 = 0.65  # Boost low energy periods
energy_component = 0.65 * 0.5 = 0.325
→ "Low-cognitive task during low energy (good fit)"
```

---

## 📈 **Test Results**

### **Test Case 1: Deep Work Coding Session**
```
Task Characteristics:
- Cognitive Load: 0.9 (Very High)
- Typical Duration: 2.0 hours
- Recovery Hours: 1.0 hour

User Energy Pattern:
- Morning (8-11): 0.9 (High)
- Afternoon (12-17): 0.7 (Medium)  
- Evening (18-23): 0.3 (Low)

Result:
✅ Scheduled: Monday 09:00
📊 Priority Score: 0.900
💡 Reasoning: "High-cognitive task • High energy period • Good cognitive-energy match"
```

### **Test Case 2: Email Processing**
```
Task Characteristics:
- Cognitive Load: 0.2 (Low)
- Typical Duration: 0.5 hours
- Recovery Hours: 0.0 hours

Expected Behavior:
- Should prefer lower energy periods
- Can be scheduled during afternoon/evening
- Less dependency on peak energy times
```

---

## 🔧 **API Integration**

### **Enhanced Endpoint Usage**
```python
# Fetch user energy pattern (168-element array)
user_energy_pattern = await _get_user_energy_pattern(user_id)

# Enhanced behavioral scheduling
result = await scheduler_service.schedule_with_behavioral_patterns(
    user_id=user_id,
    user_energy_pattern=user_energy_pattern,
    request=request,
    existing_events=existing_events,
    available_periods=None,  # Use default 7-day window
    openai_client=None,      # Will trigger LLM fallback if needed
    memory_service=None      # TODO: Add memory service
)
```

### **Response Enhancement**
```json
{
  "event": { ... },
  "scheduling_method": "enhanced_behavioral_patterns",
  "scoring_factors": {
    "habit_patterns": true,
    "energy_cognitive_fit": true
  },
  "task_type_used": {
    "cognitive_load": 0.9,
    "typical_duration": 2.0,
    "similarity_score": 0.9
  },
  "optimal_slot": {
    "score": 0.900,
    "reasoning": "High-cognitive task • High energy period • Good cognitive-energy match"
  }
}
```

---

## 🎛️ **Configuration & Tuning**

### **Weight Adjustments**
```python
# Current weights (can be made configurable)
HABIT_WEIGHT = 0.4        # Task-specific time preferences  
ENERGY_WEIGHT = 0.4       # User energy levels
COGNITIVE_WEIGHT = 0.2    # Cognitive load matching

# Alternative configurations for different users:
# Focus-oriented: (0.3, 0.5, 0.2) - Prioritize energy
# Habit-oriented: (0.6, 0.3, 0.1) - Prioritize habits  
# Balanced: (0.4, 0.4, 0.2) - Current default
```

### **Penalty Thresholds**
```python
# Duration deviation thresholds
SIGNIFICANT_DEVIATION = 2.0  # or 0.5
MODERATE_DEVIATION = 1.5     # or 0.75

# Cognitive-energy mismatch thresholds  
HIGH_COGNITIVE_THRESHOLD = 0.7
LOW_COGNITIVE_THRESHOLD = 0.3
LOW_ENERGY_PENALTY_THRESHOLD = 0.4
```

---

## 🚀 **Future Enhancements**

### **1. Dynamic Weight Learning**
```python
# Learn optimal weights per user over time
user_weights = {
    'habit_weight': 0.45,     # Learned from user feedback
    'energy_weight': 0.35,    # Adjusted based on completion rates
    'cognitive_weight': 0.20  # Tuned from success ratings
}
```

### **2. Context-Aware Scheduling**
```python
# Consider additional factors
context_factors = {
    'weather': 0.8,           # Sunny vs rainy day energy
    'day_of_week': 0.9,       # Monday vs Friday motivation  
    'recent_workload': 0.6,   # Accumulated fatigue
    'upcoming_deadlines': 0.7 # Urgency influence
}
```

### **3. Multi-User Pattern Learning**
```python
# Anonymized pattern sharing for new users
similar_user_patterns = find_users_with_similar_roles_and_habits(user)
bootstrap_patterns = aggregate_anonymous_patterns(similar_user_patterns)
```

---

## 📋 **Implementation Checklist**

- ✅ **Renamed method**: `schedule_with_habit()` → `schedule_with_behavioral_patterns()`
- ✅ **Enhanced scoring**: Added energy patterns, cognitive matching, duration optimization
- ✅ **New core method**: `find_optimal_slot_with_energy()` 
- ✅ **Improved reasoning**: `generate_enhanced_slot_reasoning()`
- ✅ **API integration**: Updated `/api/chat/message` endpoint
- ✅ **Comprehensive testing**: Created test suite with cognitive load scenarios
- ✅ **Fallback handling**: Graceful degradation to basic scheduling
- ✅ **Documentation**: Complete system architecture documentation

---

## 💡 **Key Benefits**

1. **🎯 Precision**: Multi-factor analysis for optimal time slot selection
2. **🧠 Intelligence**: Cognitive load matching prevents energy misallocation  
3. **⚖️ Balance**: Considers both learned habits and physiological energy
4. **📏 Optimization**: Duration-aware scheduling respects task nature
5. **🔄 Adaptability**: Graceful fallback when behavioral data insufficient
6. **📊 Transparency**: Rich reasoning explains scheduling decisions
7. **🚀 Scalability**: Foundation for advanced AI-driven scheduling features

The Enhanced Behavioral Scheduling System represents a significant advancement in intelligent time management, providing users with scheduling decisions that align with their natural energy rhythms, learned preferences, and task characteristics. 