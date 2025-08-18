-- Smart Scheduler Supabase Schema (Merged + Sync-Ready)
-- Minimal, normalized, iCalUID-first; safe to run multiple times

-- 1) Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Users (Expanded for structured onboarding data)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    username TEXT,
    name TEXT,
    password_hash TEXT,
    
    -- Onboarding Data
    role TEXT CHECK (role IN ('student','pm','developer','executive','teacher','engineer','other')),
    major TEXT,
    education_level TEXT,
    goals JSONB DEFAULT '[]',
    arrange_preferences JSONB DEFAULT '[]', -- From "What matters to you?"
    frequency_preferences JSONB DEFAULT '[]',
    wake_time TIME,
    sleep_time TIME,
    task_preference TEXT CHECK (task_preference IN ('single','multi')) DEFAULT 'single',

    -- Permissions & Agreements
    provider TEXT CHECK (provider IN ('google','outlook','none')) DEFAULT 'none', -- Primary calendar source
    permission_gmail BOOLEAN DEFAULT FALSE,
    permission_slack BOOLEAN DEFAULT TRUE,
    agreement_terms BOOLEAN DEFAULT FALSE,
    agreement_privacy BOOLEAN DEFAULT FALSE,
    agreement_cookie BOOLEAN DEFAULT FALSE,
    agreement_personalized BOOLEAN DEFAULT FALSE,

    -- System & Other
    timezone TEXT DEFAULT 'UTC', -- IANA 格式（例如 'Asia/Shanghai'），默认 UTC
    weekly_energy_pattern JSONB DEFAULT '[]',
    merge_strategy TEXT DEFAULT 'dedup_by_icaluid', -- 合并策略
    preferences JSONB DEFAULT '{}', -- For unstructured or legacy data, and onboarding completion flag
    scopes JSONB DEFAULT '[]',
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3) Task Types (unchanged, adds vector index below)
CREATE TABLE IF NOT EXISTS task_types (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    task_type TEXT NOT NULL,
    description TEXT,
    weekly_habit_scores JSONB DEFAULT '[]',
    slot_confidence JSONB DEFAULT '[]',
    completion_count INTEGER DEFAULT 0,
    completions_since_last_update INTEGER DEFAULT 0,
    last_mem0_update TIMESTAMPTZ,
    typical_duration FLOAT DEFAULT 1.0,
    importance_score FLOAT DEFAULT 0.5,
    recovery_hours FLOAT DEFAULT 0.5,
    cognitive_load FLOAT DEFAULT 0.5,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id,task_type)
);

-- 4) Unified Events (merged from Firestore fields + new SQL fields)
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    task_type_id UUID REFERENCES task_types(id),

    -- Core
    title TEXT NOT NULL,
    description TEXT,
    scheduled_start TIMESTAMPTZ NOT NULL, -- Firestore:start_time
    scheduled_end TIMESTAMPTZ NOT NULL,   -- Firestore:end_time 
    timezone TEXT DEFAULT 'UTC',          -- Firestore:timezone
    location TEXT,
    is_all_day BOOLEAN DEFAULT FALSE,     -- Firestore:is_all_day
    color_id TEXT,                        -- Firestore:color_id (Google)
    recurrence JSONB,                     -- Firestore:recurrence
    reminders JSONB,                      -- Firestore:reminders
    metadata JSONB DEFAULT '{}',          -- Firestore:metadata

    -- Status & Classification
    lifecycle_status TEXT DEFAULT 'scheduled', -- 业务生命周期: scheduled/in_progress/completed/canceled/delayed
    sync_status TEXT DEFAULT 'pending_create', -- 同步状态: pending_create/synced/pending_update/pending_delete/error
    category TEXT,                        -- Firestore:category
    importance_score DECIMAL(3,2) DEFAULT 0.50, -- SQL:importance_score
    fit_score DECIMAL(3,2) DEFAULT 0.50,        -- SQL:fit_score
    
    -- Type and Scheduling
    item_type TEXT CHECK (item_type IN ('task', 'event')) DEFAULT 'task',
    auto_schedule BOOLEAN DEFAULT TRUE,

    -- Planning
    deadline TIMESTAMPTZ,                 -- SQL:deadline
    alternative_slots JSONB DEFAULT '[]', -- SQL:alternative_slots
    assigned_to JSONB DEFAULT '[]',       -- Firestore:assigned_to
    emails JSONB,                         -- 参与者/联系人邮箱列表（可空）
    is_conference BOOLEAN DEFAULT FALSE,  -- 是否会议/含会议链接
    dependencies JSONB DEFAULT '[]',      -- Firestore:dependencies

    -- Merge & Sync
    ical_uid TEXT,                        -- 用于跨提供方去重
    dedup_key TEXT,                       -- hash(ical_uid+start)
    source_preferred TEXT CHECK (source_preferred IN ('google','outlook')),
    last_write_ts TIMESTAMPTZ DEFAULT NOW(),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON COLUMN events.alternative_slots IS 'Top-N候选时间段集合，兼容现有字段';

-- 5) Provider Event Mapping (Google/Outlook 双向映射)
CREATE TABLE IF NOT EXISTS event_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT CHECK (provider IN ('google','outlook')) NOT NULL,
    calendar_id TEXT,
    provider_event_id TEXT,
    ical_uid TEXT,
    etag TEXT,            -- Google
    change_key TEXT,      -- Outlook
    html_link TEXT,
    raw JSONB,            -- 原始事件JSON
    source_status TEXT,   -- active/deleted/canceled
    last_synced_at TIMESTAMPTZ,
    UNIQUE(user_id,provider,provider_event_id)
);

-- 6) Provider Calendars
CREATE TABLE IF NOT EXISTS provider_calendars (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT CHECK (provider IN ('google','outlook')) NOT NULL,
    calendar_id TEXT NOT NULL,
    display_name TEXT,
    time_zone TEXT,
    is_primary BOOLEAN DEFAULT FALSE,
    selected BOOLEAN DEFAULT TRUE,
    UNIQUE(user_id,provider,calendar_id)
);

-- 7) Sync State (per-user, per-provider)
CREATE TABLE IF NOT EXISTS sync_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT CHECK (provider IN ('google','outlook')) NOT NULL,
    sync_token TEXT,     -- Google
    delta_link TEXT,     -- Outlook
    channel_id TEXT,     -- webhook channel
    resource_id TEXT,    -- watched resource id
    expiration TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    error_count INT DEFAULT 0,
    UNIQUE(user_id,provider)
);

-- 8) Indexes
CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, scheduled_start);
CREATE INDEX IF NOT EXISTS idx_events_ical ON events(user_id, ical_uid);
CREATE INDEX IF NOT EXISTS idx_events_dedup ON events(user_id, dedup_key);
CREATE INDEX IF NOT EXISTS idx_events_item_type ON events(user_id, item_type);
CREATE INDEX IF NOT EXISTS idx_events_auto_schedule ON events(user_id, auto_schedule) WHERE auto_schedule = TRUE;
CREATE INDEX IF NOT EXISTS idx_event_sources_user_provider ON event_sources(user_id, provider);
CREATE INDEX IF NOT EXISTS idx_event_sources_ical ON event_sources(user_id, ical_uid);
CREATE INDEX IF NOT EXISTS idx_task_types_user ON task_types(user_id);
CREATE INDEX IF NOT EXISTS idx_task_types_completion_count ON task_types(completion_count);
CREATE INDEX IF NOT EXISTS idx_events_alternative_slots ON events USING GIN (alternative_slots);
CREATE INDEX IF NOT EXISTS idx_task_types_embedding ON task_types USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 9) RLS (prototype: relaxed -> 可按需收紧)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE task_types ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE event_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_calendars ENABLE ROW LEVEL SECURITY;
ALTER TABLE sync_state ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users own users" ON users;
DROP POLICY IF EXISTS "Users can insert" ON users;
DROP POLICY IF EXISTS "Users update self" ON users;
DROP POLICY IF EXISTS "Users own task_types" ON task_types;
DROP POLICY IF EXISTS "Users own events" ON events;
DROP POLICY IF EXISTS "Users own event_sources" ON event_sources;
DROP POLICY IF EXISTS "Users own provider_calendars" ON provider_calendars;
DROP POLICY IF EXISTS "Users own sync_state" ON sync_state;

-- 基础策略：允许后端服务角色(绕过RLS)，客户端RLS按user_id收敛
CREATE POLICY "Users own users" ON users FOR SELECT USING (TRUE);
CREATE POLICY "Users can insert" ON users FOR INSERT WITH CHECK (TRUE);
CREATE POLICY "Users update self" ON users FOR UPDATE USING (TRUE);

CREATE POLICY "Tasks by owner" ON task_types FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Tasks modify by owner" ON task_types FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "Tasks modify by owner2" ON task_types FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY "Events by owner" ON events FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Events insert by owner" ON events FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "Events update by owner" ON events FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY "Event sources by owner" ON event_sources FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Event sources write by owner" ON event_sources FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "Event sources update by owner" ON event_sources FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY "Calendars by owner" ON provider_calendars FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Calendars write by owner" ON provider_calendars FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "Calendars update by owner" ON provider_calendars FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY "Sync state by owner" ON sync_state FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Sync state write by owner" ON sync_state FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "Sync state update by owner" ON sync_state FOR UPDATE USING (user_id = auth.uid());

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

-- 10) Helpers
CREATE OR REPLACE FUNCTION initialize_neutral_weekly_habit_array() RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE a jsonb; i int; BEGIN a='[]'::jsonb; FOR i IN 0..167 LOOP a:=a||jsonb_build_array(0.0); END LOOP; RETURN a; END; $$;
CREATE OR REPLACE FUNCTION initialize_neutral_weekly_energy() RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE a jsonb; i int; BEGIN a='[]'::jsonb; FOR i IN 0..167 LOOP a:=a||jsonb_build_array(0.5); END LOOP; RETURN a; END; $$;
CREATE OR REPLACE FUNCTION initialize_slot_confidence() RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE m jsonb; d int; h int; a jsonb; BEGIN m='[]'::jsonb; FOR d IN 0..6 LOOP a='[]'::jsonb; FOR h IN 0..23 LOOP a:=a||jsonb_build_array(0.0); END LOOP; m:=m||jsonb_build_array(a); END LOOP; RETURN m; END; $$;

-- 11) Outbound sync helpers (MVP): BEFORE/AFTER triggers + RPC
-- BEFORE: set pending_* on INSERT/UPDATE and stamp updated_at/last_write_ts
CREATE OR REPLACE FUNCTION events_set_pending() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP='INSERT' THEN
    IF NEW.source_preferred IN ('google','outlook') THEN NEW.sync_status:='pending_create'; NEW.updated_at:=NOW(); NEW.last_write_ts:=NOW(); END IF; RETURN NEW;
  ELSIF TG_OP='UPDATE' THEN
    IF NEW.source_preferred IN ('google','outlook') AND (
      (OLD.title IS DISTINCT FROM NEW.title) OR
      (OLD.scheduled_start IS DISTINCT FROM NEW.scheduled_start) OR
      (OLD.scheduled_end   IS DISTINCT FROM NEW.scheduled_end) OR
      (OLD.description     IS DISTINCT FROM NEW.description) OR
      (OLD.location        IS DISTINCT FROM NEW.location)
    ) THEN NEW.sync_status:='pending_update'; NEW.updated_at:=NOW(); NEW.last_write_ts:=NOW(); END IF; RETURN NEW;
  END IF; RETURN NEW;
END; $$;

DROP TRIGGER IF EXISTS trg_events_set_pending ON events;
CREATE TRIGGER trg_events_set_pending BEFORE INSERT OR UPDATE ON events
FOR EACH ROW EXECUTE FUNCTION events_set_pending();

-- AFTER: notify backend via pg_net
CREATE EXTENSION IF NOT EXISTS pg_net;
CREATE OR REPLACE FUNCTION notify_sync_needed() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE v_action text; v_pid text; v_provider text;
BEGIN
  v_provider := COALESCE(NEW.source_preferred, OLD.source_preferred);
  IF TG_OP='INSERT' THEN IF v_provider IN ('google','outlook') THEN v_action:='pending_create'; END IF;
  ELSIF TG_OP='UPDATE' THEN IF v_provider IN ('google','outlook') AND NEW.sync_status='pending_update' THEN v_action:='pending_update'; END IF;
  ELSIF TG_OP='DELETE' THEN IF v_provider IN ('google','outlook') THEN v_action:='pending_delete'; END IF;
  END IF;
  IF v_action IS NULL THEN IF TG_OP='DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF; END IF;

  BEGIN
    SELECT provider_event_id INTO v_pid FROM event_sources 
      WHERE event_id=COALESCE(NEW.id,OLD.id) AND provider=v_provider LIMIT 1;
  EXCEPTION WHEN OTHERS THEN v_pid:=NULL; END;

  PERFORM net.http_post(
    url := COALESCE(current_setting('app.sync_endpoint', true),'https://www.canlah.ai/api/sync/internal/sync-trigger'),
    headers := jsonb_build_object('x-internal-key', COALESCE(current_setting('app.sync_secret', true),'change-me'),'content-type','application/json'),
    body := jsonb_build_object('event_id',COALESCE(NEW.id,OLD.id),'user_id',COALESCE(NEW.user_id,OLD.user_id),'provider',v_provider,'action',v_action,'provider_event_id',v_pid,'ts',NOW())::text
  );
  IF TG_OP='DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END; $$;

DROP TRIGGER IF EXISTS trg_sync_all ON events;
CREATE TRIGGER trg_sync_all AFTER INSERT OR UPDATE OR DELETE ON events
FOR EACH ROW EXECUTE FUNCTION notify_sync_needed();

create table if not exists used_auth_codes (
      id text primary key,
      used_at timestamptz,
      expires_at timestamptz
    );

-- RPC: mark event as synced (callable by anon)
CREATE OR REPLACE FUNCTION mark_event_synced(p_event_id uuid) RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  UPDATE events SET sync_status='synced', updated_at=NOW() WHERE id=p_event_id;
END; $$;
GRANT EXECUTE ON FUNCTION mark_event_synced(uuid) TO anon;