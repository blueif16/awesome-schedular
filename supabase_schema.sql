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
    created_at TIMESTAMP DEFAULT NOW()
);

-- Task Types (Tier 2 - Core Learning Layer)
CREATE TABLE IF NOT EXISTS task_types (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Identity
    task_type TEXT NOT NULL,
    category TEXT CHECK (category IN ('focused', 'collaborative', 'administrative')),
    
    -- Learned patterns (24-hour arrays: indices 0-23 represent hours 00:00-23:59)
    hourly_scores JSONB DEFAULT '[]',        -- User's preference for this task at each hour
    confidence_scores JSONB DEFAULT '[]',    -- Confidence in the pattern (0-1)
    performance_by_hour JSONB DEFAULT '[]', -- Energy/success levels by hour
    
    -- Task characteristics  
    cognitive_load FLOAT DEFAULT 0.5,       -- How mentally demanding (0-1)
    recovery_hours FLOAT DEFAULT 0.5,       -- Hours needed to recover after
    typical_duration FLOAT DEFAULT 1.0,     -- Average duration in hours
    importance_score FLOAT DEFAULT 0.5,     -- Learned importance (0-1)
    
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
CREATE INDEX IF NOT EXISTS idx_task_types_category ON task_types(user_id, category);

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
    category text,
    hourly_scores jsonb,
    confidence_scores jsonb,
    performance_by_hour jsonb,
    cognitive_load float,
    recovery_hours float,
    typical_duration float,
    importance_score float,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tt.id,
        tt.task_type,
        tt.category,
        tt.hourly_scores,
        tt.confidence_scores,
        tt.performance_by_hour,
        tt.cognitive_load,
        tt.recovery_hours,
        tt.typical_duration,
        tt.importance_score,
        1 - (tt.embedding <=> query_embedding) as similarity
    FROM task_types tt
    WHERE (target_user_id IS NULL OR tt.user_id = target_user_id)
        AND 1 - (tt.embedding <=> query_embedding) > match_threshold
    ORDER BY tt.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Helper function to initialize neutral 24-hour arrays
CREATE OR REPLACE FUNCTION initialize_neutral_arrays()
RETURNS jsonb[]
LANGUAGE plpgsql
AS $$
DECLARE
    hourly_scores jsonb;
    confidence_scores jsonb;
    performance_scores jsonb;
BEGIN
    -- Initialize all hours with neutral values
    hourly_scores := jsonb_build_array(
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  -- 00-05: Neutral preference
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  -- 06-11: Neutral preference  
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  -- 12-17: Neutral preference
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5   -- 18-23: Neutral preference
    );
    
    confidence_scores := jsonb_build_array(
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  -- 00-05: Low confidence initially
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  -- 06-11: Low confidence initially
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  -- 12-17: Low confidence initially
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1   -- 18-23: Low confidence initially
    );
    
    performance_scores := jsonb_build_array(
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  -- 00-05: Neutral performance
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  -- 06-11: Neutral performance
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  -- 12-17: Neutral performance
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5   -- 18-23: Neutral performance
    );
    
    RETURN ARRAY[hourly_scores, confidence_scores, performance_scores];
END;
$$; 