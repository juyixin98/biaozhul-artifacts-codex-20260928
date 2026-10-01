CREATE SCHEMA IF NOT EXISTS diststats_b;

CREATE TABLE IF NOT EXISTS diststats_b.jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    estimator TEXT NOT NULL CHECK (estimator IN ('population', 'sample')),
    expected_shards INTEGER NOT NULL CHECK (expected_shards >= 0),
    status TEXT NOT NULL CHECK (status IN ('open', 'complete')),
    received_shards INTEGER NOT NULL DEFAULT 0 CHECK (received_shards >= 0),
    current_summary JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS diststats_b.shards (
    job_id TEXT NOT NULL REFERENCES diststats_b.jobs(id) ON DELETE CASCADE,
    shard_id TEXT NOT NULL,
    sequence BIGINT NOT NULL,
    producer_id TEXT NOT NULL DEFAULT '',
    observations INTEGER NOT NULL CHECK (observations >= 0),
    weight DOUBLE PRECISION NOT NULL CHECK (weight >= 0),
    summary JSONB NOT NULL,
    payload JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, shard_id),
    UNIQUE (job_id, sequence)
);

CREATE TABLE IF NOT EXISTS diststats_b.events (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES diststats_b.jobs(id) ON DELETE CASCADE,
    sequence BIGINT NOT NULL,
    run_id TEXT NOT NULL,
    shard_id TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    error_class TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    request_json JSONB,
    before_state JSONB,
    after_state JSONB,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, sequence)
);

CREATE INDEX IF NOT EXISTS events_job_sequence_idx ON diststats_b.events (job_id, sequence);
CREATE INDEX IF NOT EXISTS shards_job_sequence_idx ON diststats_b.shards (job_id, sequence);
