CREATE TABLE IF NOT EXISTS generations (
  generation text PRIMARY KEY,
  created_at timestamptz NOT NULL DEFAULT now(),
  description text NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ranked_items (
  generation text NOT NULL REFERENCES generations(generation) ON DELETE CASCADE,
  shard_id text NOT NULL,
  item_id text NOT NULL,
  score bigint NOT NULL,
  label text NOT NULL,
  PRIMARY KEY (generation, item_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS ranked_items_global_unique
  ON ranked_items (generation, item_id);

CREATE INDEX IF NOT EXISTS ranked_items_ordered
  ON ranked_items (generation, shard_id, score DESC, item_id ASC);

CREATE TABLE IF NOT EXISTS query_runs (
  run_id text PRIMARY KEY,
  generation text NOT NULL,
  k integer NOT NULL CHECK (k > 0),
  complete_flag boolean NOT NULL,
  exact_flag boolean NOT NULL,
  result_count integer NOT NULL,
  stop_reason text NOT NULL,
  bound_probes bigint NOT NULL DEFAULT 0,
  storage_rows_read bigint NOT NULL DEFAULT 0,
  candidates_returned bigint NOT NULL DEFAULT 0,
  candidates_considered bigint NOT NULL DEFAULT 0,
  error_summary text,
  created_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS query_events (
  run_id text NOT NULL REFERENCES query_runs(run_id) ON DELETE CASCADE,
  sequence bigint NOT NULL,
  event_type text NOT NULL,
  shard_id text,
  generation text NOT NULL,
  complete_flag boolean NOT NULL,
  exact_flag boolean NOT NULL,
  heap_size integer NOT NULL,
  threshold_score bigint,
  threshold_item_id text,
  upper_score bigint,
  upper_item_id text,
  cursor_exhausted boolean NOT NULL,
  pages_scanned bigint NOT NULL,
  bound_probes bigint NOT NULL,
  rows_read bigint NOT NULL,
  failure_category text,
  message text,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS query_events_run_sequence
  ON query_events(run_id, sequence);
