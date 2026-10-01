ALTER TABLE diststats_b.events
ADD COLUMN IF NOT EXISTS request_json JSONB;

