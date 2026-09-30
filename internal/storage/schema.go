package storage

import "context"

const schemaSQL = `
CREATE TABLE IF NOT EXISTS generations (
  generation_id TEXT PRIMARY KEY,
  item_count BIGINT NOT NULL CHECK (item_count >= 0),
  checksum TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS items (
  generation_id TEXT NOT NULL REFERENCES generations(generation_id),
  shard_id INTEGER NOT NULL CHECK (shard_id >= 0),
  item_id TEXT NOT NULL,
  score BIGINT NOT NULL,
  sensitive_label TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (generation_id, item_id),
  UNIQUE (generation_id, shard_id, item_id)
);

CREATE INDEX IF NOT EXISTS items_shard_order_idx
  ON items (generation_id, shard_id, score DESC, item_id ASC);

CREATE TABLE IF NOT EXISTS topk_queries (
  request_id TEXT PRIMARY KEY,
  generation_id TEXT NOT NULL REFERENCES generations(generation_id),
  k INTEGER NOT NULL CHECK (k > 0),
  exact BOOLEAN NOT NULL,
  complete BOOLEAN NOT NULL,
  failure_class TEXT,
  failure_message TEXT,
  threshold JSONB,
  rounds INTEGER NOT NULL,
  total_rows_read INTEGER NOT NULL,
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS topk_query_events (
  request_id TEXT NOT NULL REFERENCES topk_queries(request_id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  shard_id INTEGER,
  round INTEGER NOT NULL,
  cursor_key JSONB,
  upper_bound_key JSONB,
  threshold_key JSONB,
  item_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
  reason TEXT NOT NULL,
  failure_class TEXT,
  rows_read INTEGER NOT NULL,
  occurred_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (request_id, seq)
);

CREATE TABLE IF NOT EXISTS topk_query_results (
  request_id TEXT NOT NULL REFERENCES topk_queries(request_id) ON DELETE CASCADE,
  position INTEGER NOT NULL CHECK (position > 0),
  item_id TEXT NOT NULL,
  score BIGINT NOT NULL,
  PRIMARY KEY (request_id, position)
);

CREATE TABLE IF NOT EXISTS topk_query_shard_states (
  request_id TEXT NOT NULL REFERENCES topk_queries(request_id) ON DELETE CASCADE,
  shard_id INTEGER NOT NULL,
  exhausted BOOLEAN NOT NULL,
  active BOOLEAN NOT NULL,
  upper_bound_key JSONB,
  cursor_key JSONB,
  pages INTEGER NOT NULL,
  rows_read INTEGER NOT NULL,
  PRIMARY KEY (request_id, shard_id)
);
`

func (s *Store) EnsureSchema(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, schemaSQL)
	return err
}
