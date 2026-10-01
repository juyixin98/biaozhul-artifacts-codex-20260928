CREATE TABLE IF NOT EXISTS checkpoints (
    task_id TEXT NOT NULL,
    id BIGINT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    state JSONB NOT NULL,
    inflight JSONB NOT NULL,
    positions JSONB NOT NULL,
    metrics JSONB NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (task_id, id)
);

ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS metrics JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS checkpoints_monotonic_complete
    ON checkpoints (task_id, id)
    WHERE status = 'complete';

CREATE TABLE IF NOT EXISTS checkpoint_outputs (
    task_id TEXT NOT NULL,
    checkpoint_id BIGINT NOT NULL,
    output_id TEXT NOT NULL,
    stream TEXT NOT NULL,
    key TEXT NOT NULL,
    seq BIGINT NOT NULL,
    epoch BIGINT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (task_id, output_id),
    FOREIGN KEY (task_id, checkpoint_id) REFERENCES checkpoints(task_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS checkpoint_outputs_checkpoint
    ON checkpoint_outputs (task_id, checkpoint_id);
