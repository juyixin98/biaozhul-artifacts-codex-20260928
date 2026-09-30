package state

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"topk-demo/internal/protocol"
)

type PostgresStore struct {
	pool *pgxpool.Pool
}

func NewPostgresStore(ctx context.Context, dsn string) (*PostgresStore, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return &PostgresStore{pool: pool}, nil
}

func (s *PostgresStore) Close() { s.pool.Close() }

func (s *PostgresStore) SaveRun(ctx context.Context, run protocol.RunRecord) error {
	_, err := s.pool.Exec(ctx, `
INSERT INTO query_runs
(run_id, generation, k, complete_flag, exact_flag, result_count, stop_reason,
 bound_probes, storage_rows_read, candidates_returned, candidates_considered,
 error_summary, created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
ON CONFLICT (run_id) DO UPDATE SET
 complete_flag = EXCLUDED.complete_flag,
 exact_flag = EXCLUDED.exact_flag,
 result_count = EXCLUDED.result_count,
 stop_reason = EXCLUDED.stop_reason,
 bound_probes = EXCLUDED.bound_probes,
 storage_rows_read = EXCLUDED.storage_rows_read,
 candidates_returned = EXCLUDED.candidates_returned,
 candidates_considered = EXCLUDED.candidates_considered,
 error_summary = EXCLUDED.error_summary`,
		run.RunID, run.Generation, run.K, run.Complete, run.Exact, run.ResultCount,
		run.StopReason, run.BoundProbes, run.StorageRowsRead, run.CandidatesReturned,
		run.CandidatesConsidered, run.ErrorSummary, run.CreatedAt)
	return err
}

func (s *PostgresStore) SaveEvent(ctx context.Context, ev protocol.EventRecord) error {
	_, err := s.pool.Exec(ctx, `
INSERT INTO query_events
(run_id, sequence, event_type, shard_id, generation, complete_flag, exact_flag,
 heap_size, threshold_score, threshold_item_id, upper_score, upper_item_id,
 cursor_exhausted, pages_scanned, bound_probes, rows_read, failure_category,
 message, created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
ON CONFLICT (run_id, sequence) DO NOTHING`,
		ev.RunID, ev.Sequence, ev.EventType, nullable(ev.ShardID), ev.Generation,
		ev.Complete, ev.Exact, ev.HeapSize, ev.ThresholdScore, nullable(ev.ThresholdItemID),
		ev.UpperScore, nullable(ev.UpperItemID), ev.CursorExhausted, ev.PagesScanned,
		ev.BoundProbes, ev.RowsRead, nullable(ev.FailureCategory), nullable(ev.Message), ev.CreatedAt)
	return err
}

func (s *PostgresStore) GetRun(ctx context.Context, runID string) (*protocol.RunRecord, []protocol.EventRecord, error) {
	var run protocol.RunRecord
	err := s.pool.QueryRow(ctx, `
SELECT run_id, generation, k, complete_flag, exact_flag, result_count, stop_reason,
       bound_probes, storage_rows_read, candidates_returned, candidates_considered,
       COALESCE(error_summary, ''), created_at
FROM query_runs WHERE run_id = $1`, runID).Scan(
		&run.RunID, &run.Generation, &run.K, &run.Complete, &run.Exact, &run.ResultCount,
		&run.StopReason, &run.BoundProbes, &run.StorageRowsRead, &run.CandidatesReturned,
		&run.CandidatesConsidered, &run.ErrorSummary, &run.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil, fmt.Errorf("run %q not found", runID)
	}
	if err != nil {
		return nil, nil, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT run_id, sequence, COALESCE(event_type,''), COALESCE(shard_id,''), generation,
       complete_flag, exact_flag, heap_size, threshold_score, COALESCE(threshold_item_id,''),
       upper_score, COALESCE(upper_item_id,''), cursor_exhausted, pages_scanned,
       bound_probes, rows_read, COALESCE(failure_category,''), COALESCE(message,''), created_at
FROM query_events WHERE run_id = $1 ORDER BY sequence`, runID)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	var events []protocol.EventRecord
	for rows.Next() {
		var ev protocol.EventRecord
		if err := rows.Scan(
			&ev.RunID, &ev.Sequence, &ev.EventType, &ev.ShardID, &ev.Generation,
			&ev.Complete, &ev.Exact, &ev.HeapSize, &ev.ThresholdScore, &ev.ThresholdItemID,
			&ev.UpperScore, &ev.UpperItemID, &ev.CursorExhausted, &ev.PagesScanned,
			&ev.BoundProbes, &ev.RowsRead, &ev.FailureCategory, &ev.Message, &ev.CreatedAt,
		); err != nil {
			return nil, nil, err
		}
		events = append(events, ev)
	}
	return &run, events, rows.Err()
}

func nullable(v string) any {
	if v == "" {
		return nil
	}
	return v
}
