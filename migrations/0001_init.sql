CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    dim INTEGER NOT NULL CHECK (dim > 0),
    status TEXT NOT NULL CHECK (status IN ('open', 'finalized')),
    aggregate JSONB NOT NULL,
    version BIGINT NOT NULL DEFAULT 0 CHECK (version >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finalized_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS shards (
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    shard_id TEXT NOT NULL,
    seq BIGINT NOT NULL,
    n BIGINT NOT NULL CHECK (n >= 0),
    weight DOUBLE PRECISION NOT NULL CHECK (weight >= 0),
    summary JSONB NOT NULL,
    submitted_run_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, shard_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS shards_job_seq_idx
    ON shards (job_id, seq);

CREATE SEQUENCE IF NOT EXISTS global_run_number_seq;

CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY,
    run_number BIGINT NOT NULL UNIQUE DEFAULT nextval('global_run_number_seq'),
    job_id UUID,
    route TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INTEGER,
    status TEXT NOT NULL CHECK (status IN ('started', 'succeeded', 'failed')),
    request JSONB,
    error JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS runs_job_number_idx ON runs (job_id, run_number);
CREATE INDEX IF NOT EXISTS runs_started_idx ON runs (started_at);

CREATE TABLE IF NOT EXISTS run_events (
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    event_index BIGINT NOT NULL,
    stage TEXT NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('info', 'succeeded', 'failed')),
    reason TEXT NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, event_index)
);

CREATE INDEX IF NOT EXISTS run_events_run_idx ON run_events (run_id, event_index);
