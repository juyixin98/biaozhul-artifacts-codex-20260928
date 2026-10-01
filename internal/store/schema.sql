CREATE TABLE IF NOT EXISTS coordinator_run (
  epoch bigint PRIMARY KEY,
  started_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  id text PRIMARY KEY,
  status text NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
  total_shards integer NOT NULL CHECK (total_shards > 0),
  max_attempts_per_shard integer NOT NULL CHECK (max_attempts_per_shard > 0),
  max_speculative_copies integer NOT NULL CHECK (max_speculative_copies >= 0),
  spec_after interval NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shards (
  job_id text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  id text NOT NULL,
  idx integer NOT NULL,
  status text NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, id)
);

CREATE TABLE IF NOT EXISTS attempts (
  job_id text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  shard_id text NOT NULL,
  attempt_no integer NOT NULL,
  run_epoch bigint NOT NULL,
  token text NOT NULL UNIQUE,
  worker_id text NOT NULL DEFAULT '',
  kind text NOT NULL CHECK (kind IN ('initial','retry','speculative')),
  status text NOT NULL CHECK (status IN ('ready','running','succeeded','failed','rejected')),
  failure_category text NOT NULL DEFAULT '',
  progress double precision NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
  started_at timestamptz NOT NULL DEFAULT now(),
  last_heartbeat_at timestamptz,
  ended_at timestamptz,
  PRIMARY KEY (job_id, shard_id, attempt_no),
  FOREIGN KEY (job_id, shard_id) REFERENCES shards(job_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS attempts_claim_idx
  ON attempts(status, job_id, shard_id, attempt_no);

CREATE TABLE IF NOT EXISTS artifacts (
  job_id text NOT NULL,
  shard_id text NOT NULL,
  attempt_no integer NOT NULL,
  content_hash text NOT NULL,
  size bigint NOT NULL,
  content bytea NOT NULL,
  published_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, shard_id),
  FOREIGN KEY (job_id, shard_id, attempt_no)
    REFERENCES attempts(job_id, shard_id, attempt_no)
);

CREATE SEQUENCE IF NOT EXISTS events_seq AS bigint;

CREATE TABLE IF NOT EXISTS events (
  seq bigint PRIMARY KEY DEFAULT nextval('events_seq'),
  job_id text NOT NULL DEFAULT '',
  shard_id text NOT NULL DEFAULT '',
  attempt_no integer NOT NULL DEFAULT 0,
  worker_id text NOT NULL DEFAULT '',
  type text NOT NULL,
  status text NOT NULL DEFAULT '',
  category text NOT NULL DEFAULT '',
  reason text NOT NULL DEFAULT '',
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id, seq);
