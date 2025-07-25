# Smart Scheduler Prototype Setup Guide

**[Nexus]** Ready to test your AI-powered learning scheduler? Let's get you set up in under 5 minutes! 🚀

## 📋 Prerequisites

You'll need:
- Python 3.9+
- A Supabase account (free tier works)
- An OpenAI API key
- Basic familiarity with command line

## 🚀 Quick Setup

### Step 1: Supabase Database Setup

1. **Create a Supabase project** at [supabase.com](https://supabase.com)
2. **Run the schema** in your Supabase SQL Editor:
   ```sql
   -- Copy and paste the entire content of supabase_schema.sql
   ```
3. **Get your credentials**:
   - Project URL: `https://your-project.supabase.co`
   - Anon Key: Found in Settings → API

### Step 2: OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Create an API key
3. You'll need ~$1-2 for testing (embeddings are cheap!)

### Step 3: Environment Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create environment file**:
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your credentials:
   OPENAI_API_KEY=your_openai_api_key_here
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your_supabase_anon_key
   ```

### Step 4: Run the Prototype

```bash
python prototype.py
```

## 🎯 Testing the System

### Phase 1: Setup
1. Run the prototype
2. Create your user profile (pick your role)
3. You're ready to test!

### Phase 2: Initial Scheduling
Test the system with neutral patterns:

```
Event: "Deep coding session"
Duration: 2 hours
Description: "Work on machine learning project"
```

**Expected**: Gets scheduled with neutral score (~0.5), creates new task type

### Phase 3: Learning Simulation
Use menu option 4 to simulate learning data. This creates:
- 📈 High scores for morning coding (8-10 AM)
- 📉 Low scores for post-lunch (1-2 PM)  
- 📊 Medium scores for evening (7-8 PM)

### Phase 4: Intelligent Scheduling
Schedule another coding session - watch it prefer morning slots now!

```
Event: "Fix authentication bug"
Duration: 1 hour
Description: "Debug OAuth integration"
```

**Expected**: High similarity match to "Deep coding session", prefers morning

### Phase 5: Manual Learning
Complete events with feedback:
- High energy (0.8-0.9) for morning tasks
- Low energy (0.3-0.4) for afternoon tasks
- Watch patterns update in real-time!

## 🔍 What to Watch For

### ✅ Tier 2 Learning in Action

**After simulation, you should see:**
```
🎯 Task Type: Deep Coding Session
   Category: focused
   Cognitive Load: 0.8
   🎯 Best times for 'Deep Coding Session': 8:00, 9:00, 10:00
   ❌ Avoid 'Deep Coding Session' at: 13:00, 14:00
   ⏰ You prefer 'Deep Coding Session' in the morning
   📊 High confidence in these patterns (75%)
```

### ✅ Vector Similarity Working

When scheduling similar tasks:
```
📊 Found similar task: 'Deep Coding Session' (similarity: 0.92)
```

### ✅ Smart Scheduling Decisions

**Morning slot selection:**
```
✅ EVENT SCHEDULED SUCCESSFULLY!
📅 Time: Monday, December 16 at 08:00 AM
📊 Score: 0.87
💡 Reasoning: High preference for Deep Coding Session at 8:00 • High energy period • Good energy for demanding task • High confidence in pattern
```

## 🧪 Advanced Testing

### Test Vector Similarity
Try scheduling tasks with similar descriptions:
- "Code review session" → Should match "Deep coding session"
- "Debug API endpoints" → Should match "Deep coding session"  
- "Team standup meeting" → Should create new collaborative task type

### Test Category Classification
- **Focused**: "Study for ML exam", "Write research paper"
- **Collaborative**: "Sprint planning", "Client presentation"
- **Administrative**: "Expense reports", "Email cleanup"

### Test Learning Speed
- Complete tasks at different hours
- Watch confidence scores increase
- See patterns emerge after 3-5 completions

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Test your connection
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
print('✅ Supabase connected!')
"
```

### OpenAI API Issues
```bash
# Test embeddings
python -c "
import openai
import os
from dotenv import load_dotenv
load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.embeddings.create(model='text-embedding-3-small', input='test')
print('✅ OpenAI connected!')
"
```

### Schema Issues
Make sure the vector extension is enabled:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## 🎯 Success Metrics

After testing, you should see:

1. **Task Type Creation**: ✅ New task types auto-categorized
2. **Vector Similarity**: ✅ Similar tasks matched (>0.8 similarity)
3. **Pattern Learning**: ✅ Hourly scores updating after completions
4. **Smart Scheduling**: ✅ High-confidence hours preferred
5. **Reasoning**: ✅ Clear explanations for scheduling decisions

## 📊 Sample Test Sequence

Here's a complete test flow:

```bash
# 1. Start prototype
python prototype.py

# 2. Setup user (Developer role)
# 3. Schedule "Deep coding session" (2 hours)
# 4. Menu option 4: Simulate learning data
# 5. View patterns (option 3) - see morning preference
# 6. Schedule "Fix React bug" (1 hour) - should prefer morning
# 7. Complete an event with high energy (0.9)
# 8. View patterns again - see confidence increase
```

## 🚀 Next Steps

Once you've verified the prototype works:
- The system demonstrates the three-tier architecture
- Task types learn from completions  
- Vector similarity finds related tasks
- Scheduling uses learned patterns for optimal timing

Ready to see your scheduler get smarter with every task! 🧠✨ 