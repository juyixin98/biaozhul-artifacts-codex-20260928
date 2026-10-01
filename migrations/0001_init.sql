CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'succeeded', 'failed')),
    version INTEGER NOT NULL DEFAULT 0,
    total_shards INTEGER NOT NULL CHECK (total_shards > 0),
    max_attempts_per_shard INTEGER NOT NULL,
    max_active_attempts INTEGER NOT NULL,
    lease_ms INTEGER NOT NULL,
    speculate_after_ms INTEGER NOT NULL,
    heartbeat_interval_ms INTEGER NOT NULL,
    failure_category TEXT NOT NULL DEFAULT '',
    failure_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shards (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    shard_key TEXT NOT NULL,
    input TEXT NOT NULL,
    behavior_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'succeeded', 'failed')),
    attempt_number INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    failure_category TEXT NOT NULL DEFAULT '',
    failure_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, ordinal),
    UNIQUE (job_id, shard_key)
);

CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    shard_id TEXT NOT NULL REFERENCES shards(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('original', 'speculative')),
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'succeeded', 'failed', 'expired', 'interrupted', 'superseded')),
    epoch TEXT NOT NULL,
    worker_url TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    failure_category TEXT NOT NULL DEFAULT '',
    failure_message TEXT NOT NULL DEFAULT '',
    artifact_id TEXT,
    checksum_sha256 TEXT NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL DEFAULT 0,
    lease_expires_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (shard_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    shard_id TEXT NOT NULL REFERENCES shards(id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS artifacts_one_published_shard
    ON artifacts(shard_id);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    shard_id TEXT NOT NULL DEFAULT '',
    attempt_id TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    epoch TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_shards_job_ordinal ON shards(job_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_attempts_job_state ON attempts(job_id, state);
CREATE INDEX IF NOT EXISTS idx_attempts_schedule ON attempts(state, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_events_job_id ON events(job_id, id);
CREATE INDEX IF NOT EXISTS idx_shards_job_state ON shards(job_id, state);
