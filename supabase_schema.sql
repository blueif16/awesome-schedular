-- Smart Scheduler Supabase Schema
-- Tier 2: Task Types Layer Implementation

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table (simplified for prototype)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    role TEXT CHECK (role IN ('student', 'pm', 'developer', 'executive')),
    timezone TEXT DEFAULT 'UTC',
    
    -- Weekly energy pattern (168 elements: 24 hours × 7 days)
    -- Index formula: day_of_week * 24 + hour (0=Sunday 00:00, 167=Saturday 23:59)
    -- Values 0.0-1.0: User's natural energy levels by hour across the week
    weekly_energy_pattern JSONB DEFAULT '[]',
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Task Types (Tier 2 - Core Learning Layer)
CREATE TABLE IF NOT EXISTS task_types (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Identity
    task_type TEXT NOT NULL,
    description TEXT,                        -- Optional description for better embedding
    
    -- Learned patterns (168-hour weekly array: 24 hours × 7 days)
    -- Index formula: day_of_week * 24 + hour (0=Sunday 00:00, 167=Saturday 23:59)
    -- Values 0.0-1.0: Task-specific success/preference patterns at each time
    weekly_habit_scores JSONB DEFAULT '[]',
    slot_confidence JSONB DEFAULT '[]',     -- 7x24 confidence matrix for each time slot
    
    completion_count INTEGER DEFAULT 0,     -- Total number of times this task type has been completed
    completions_since_last_update INTEGER DEFAULT 0, -- Completions since last mem0 update (resets to 0 after update)
    last_mem0_update TIMESTAMP,            -- Track when last updated from mem0 insights
    
    -- Task characteristics
    typical_duration FLOAT DEFAULT 1.0,     -- Average duration in hours
    importance_score FLOAT DEFAULT 0.5,     -- Learned importance from mem0 (0-1)
    recovery_hours FLOAT DEFAULT 0.5,       -- Buffer time needed after this task
    cognitive_load FLOAT DEFAULT 0.5,       -- Task mental difficulty (0-1)
    
    -- Vector for similarity matching
    embedding vector(1536),                 -- OpenAI text-embedding-3-small
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(user_id, task_type)
);

-- Events (Tier 1 - Raw scheduling data)
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    task_type_id UUID REFERENCES task_types(id),
    
    -- Event details
    title TEXT NOT NULL,
    description TEXT,
    scheduled_start TIMESTAMP NOT NULL,
    scheduled_end TIMESTAMP NOT NULL,
    deadline TIMESTAMP,                      -- For urgency calculation
    
    -- Completion tracking (for learning)
    completed BOOLEAN DEFAULT FALSE,
    success_rating FLOAT,                    -- 0-1: How well did it go?
    energy_after FLOAT,                      -- 0-1: Energy level after completion
    energy_before FLOAT,                     -- 0-1: Energy level before start
    perceived_difficulty FLOAT,              -- 0-1: How hard was it?
    actual_start TIMESTAMP,
    actual_end TIMESTAMP,
    
    -- Priority calculation
    calculated_priority FLOAT DEFAULT 0.5,  -- Final priority score
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, scheduled_start);
CREATE INDEX IF NOT EXISTS idx_task_types_user ON task_types(user_id);
CREATE INDEX IF NOT EXISTS idx_task_types_embedding ON task_types USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_task_types_completion_count ON task_types(completion_count);

-- Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE task_types ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;

-- Create policies (simplified for prototype - using user_id instead of auth.uid())
-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can access own data" ON users;
DROP POLICY IF EXISTS "Users can access own task types" ON task_types;
DROP POLICY IF EXISTS "Users can access own events" ON events;

-- Create new policies
CREATE POLICY "Users can access own data" ON users
    FOR ALL USING (TRUE);  -- Simplified for prototype

CREATE POLICY "Users can access own task types" ON task_types
    FOR ALL USING (TRUE);  -- Simplified for prototype

CREATE POLICY "Users can access own events" ON events
    FOR ALL USING (TRUE);  -- Simplified for prototype

-- Drop existing function first to avoid return type conflicts
DROP FUNCTION IF EXISTS match_task_types(vector, float, int, uuid);

-- Vector similarity search function
CREATE OR REPLACE FUNCTION match_task_types(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 1,
    target_user_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    task_type text,
    description text,
    weekly_habit_scores jsonb,
    slot_confidence jsonb,
    completion_count integer,
    completions_since_last_update integer,
    typical_duration float,
    importance_score float,
    recovery_hours float,
    cognitive_load float,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tt.id,
        tt.task_type,
        tt.description,
        tt.weekly_habit_scores,
        tt.slot_confidence,
        tt.completion_count,
        tt.completions_since_last_update,
        tt.typical_duration,
        tt.importance_score,
        tt.recovery_hours,
        tt.cognitive_load,
        1 - (tt.embedding <=> query_embedding) as similarity
    FROM task_types tt
    WHERE (target_user_id IS NULL OR tt.user_id = target_user_id)
        AND 1 - (tt.embedding <=> query_embedding) > match_threshold
    ORDER BY tt.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Helper function to initialize neutral 168-hour weekly habit array  
CREATE OR REPLACE FUNCTION initialize_neutral_weekly_habit_array()
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    weekly_habit_scores jsonb;
    i integer;
BEGIN
    -- Initialize with neutral habit pattern for all 168 hours (24 hours × 7 days)
    weekly_habit_scores := '[]'::jsonb;
    
    -- Build 168-element array (7 days × 24 hours each)
    FOR i IN 0..167 LOOP
        weekly_habit_scores := weekly_habit_scores || jsonb_build_array(0.0);
    END LOOP;
    
    RETURN weekly_habit_scores;
END;
$$;

-- Helper function to initialize neutral weekly energy pattern (168 elements)
CREATE OR REPLACE FUNCTION initialize_neutral_weekly_energy()
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    weekly_energy jsonb;
    daily_pattern jsonb;
    i integer;
BEGIN
    -- Initialize with neutral energy pattern for all 168 hours (24 hours × 7 days)
    weekly_energy := '[]'::jsonb;
    
    -- Build 168-element array (7 days × 24 hours each)
    FOR i IN 0..167 LOOP
        weekly_energy := weekly_energy || jsonb_build_array(0.5);
    END LOOP;
    
    RETURN weekly_energy;
END;
$$;

-- Helper function to initialize slot confidence matrix (7x24 matrix)
CREATE OR REPLACE FUNCTION initialize_slot_confidence()
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    confidence_matrix jsonb;
    day_array jsonb;
    day integer;
    hour integer;
BEGIN
    -- Initialize 7x24 matrix with all zeros (7 days, 24 hours each)
    confidence_matrix := '[]'::jsonb;
    
    FOR day IN 0..6 LOOP
        day_array := '[]'::jsonb;
        
        FOR hour IN 0..23 LOOP
            day_array := day_array || jsonb_build_array(0.0);
        END LOOP;
        
        confidence_matrix := confidence_matrix || jsonb_build_array(day_array);
    END LOOP;
    
    RETURN confidence_matrix;
END;
$$;