package storage

const Schema = `
CREATE TABLE IF NOT EXISTS jobs (
    id text PRIMARY KEY,
    mode text NOT NULL,
    config jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS checkpoints (
    job_id text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    epoch bigint NOT NULL,
    attempt_id text NOT NULL,
    mode text NOT NULL,
    status text NOT NULL,
    base_state jsonb NOT NULL,
    inflights jsonb NOT NULL DEFAULT '{}'::jsonb,
    targets jsonb NOT NULL,
    seen_barriers jsonb NOT NULL,
    base_offsets jsonb NOT NULL,
    version bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    sealed_at timestamptz,
    PRIMARY KEY (job_id, epoch)
);

CREATE UNIQUE INDEX IF NOT EXISTS checkpoints_one_active_per_job
    ON checkpoints(job_id)
    WHERE status = 'ACTIVE';

CREATE TABLE IF NOT EXISTS checkpoint_records (
    job_id text NOT NULL,
    checkpoint_epoch bigint NOT NULL,
    stream text NOT NULL,
    offset_position bigint NOT NULL,
    record_epoch bigint NOT NULL,
    key_name text NOT NULL,
    value_delta bigint NOT NULL,
    PRIMARY KEY (job_id, checkpoint_epoch, stream, offset_position),
    FOREIGN KEY (job_id, checkpoint_epoch) REFERENCES checkpoints(job_id, epoch) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS outputs (
    output_id text PRIMARY KEY,
    job_id text NOT NULL,
    stream text NOT NULL,
    offset_position bigint NOT NULL,
    epoch bigint NOT NULL,
    key_name text NOT NULL,
    value_delta bigint NOT NULL,
    committed boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id, stream, offset_position)
);

CREATE INDEX IF NOT EXISTS outputs_job_epoch_idx ON outputs(job_id, epoch, committed);

CREATE TABLE IF NOT EXISTS request_events (
    id bigserial PRIMARY KEY,
    job_id text NOT NULL,
    request_id text NOT NULL,
    epoch bigint NOT NULL,
    step text NOT NULL,
    location text NOT NULL,
    detail text NOT NULL,
    occurred_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS request_events_job_idx ON request_events(job_id, id);
CREATE INDEX IF NOT EXISTS request_events_request_idx ON request_events(request_id);
`
