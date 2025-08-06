-- Add alternative_slots column to events table
-- This stores top-rated alternative time slots for each event to enable fast rescheduling

ALTER TABLE events ADD COLUMN alternative_slots JSONB DEFAULT '[]';

-- Add comment to explain the column structure
COMMENT ON COLUMN events.alternative_slots IS 
'Stores top 5 alternative time slots for the event in JSON format:
[
  {
    "start": "2024-01-15T10:00:00Z",
    "end": "2024-01-15T11:00:00Z", 
    "score": 0.85,
    "fit_score": 0.9,
    "priority_score": 0.8
  }
]';

-- Add index for querying alternative slots
CREATE INDEX idx_events_alternative_slots ON events USING GIN (alternative_slots);

-- Add fit_score and priority_score columns if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='events' AND column_name='fit_score') THEN
        ALTER TABLE events ADD COLUMN fit_score DECIMAL(3,2) DEFAULT 0.5;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='events' AND column_name='priority_score') THEN
        ALTER TABLE events ADD COLUMN priority_score DECIMAL(3,2) DEFAULT 0.5;
    END IF;
END $$; 