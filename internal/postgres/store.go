package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"diststats/internal/protocol"
	"diststats/internal/stats"
)

type Store struct {
	pool *pgxpool.Pool
}

func NewStore(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool}
}

func (s *Store) CreateJob(ctx context.Context, request protocol.JobRequest) (protocol.Job, *protocol.Summary, error) {
	id, err := randomID("job")
	if err != nil {
		return protocol.Job{}, nil, err
	}
	empty, err := stats.NewSummary(request.Dimension)
	if err != nil {
		return protocol.Job{}, nil, err
	}
	summaryJSON, err := json.Marshal(empty)
	if err != nil {
		return protocol.Job{}, nil, protocol.NewServiceError(protocol.ClassStorage, "STATE_ENCODE_FAILED",
			"could not encode initial state", err)
	}

	status := "open"
	if request.ExpectedShards == 0 {
		status = "complete"
	}
	var job protocol.Job
	err = s.pool.QueryRow(ctx, `
INSERT INTO diststats_b.jobs (id, run_id, dimension, estimator, expected_shards, status, current_summary)
VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING id, run_id, dimension, estimator, expected_shards, status, received_shards, created_at, updated_at`,
		id, request.RunID, request.Dimension, request.Estimator, request.ExpectedShards, status, summaryJSON,
	).Scan(&job.ID, &job.RunID, &job.Dimension, &job.Estimator, &job.ExpectedShards, &job.Status,
		&job.ReceivedShards, &job.CreatedAt, &job.UpdatedAt)
	if err != nil {
		return protocol.Job{}, nil, mapWriteError(err)
	}
	return job, empty, nil
}

func (s *Store) GetJob(ctx context.Context, id string) (protocol.Job, *protocol.Summary, error) {
	var job protocol.Job
	var summaryJSON []byte
	err := s.pool.QueryRow(ctx, `
SELECT id, run_id, dimension, estimator, expected_shards, status, received_shards,
       current_summary, created_at, updated_at
FROM diststats_b.jobs WHERE id = $1`, id).Scan(
		&job.ID, &job.RunID, &job.Dimension, &job.Estimator, &job.ExpectedShards, &job.Status,
		&job.ReceivedShards, &summaryJSON, &job.CreatedAt, &job.UpdatedAt)
	if err != nil {
		return protocol.Job{}, nil, mapReadError(err)
	}
	summary, err := decodeSummary(summaryJSON)
	if err != nil {
		return protocol.Job{}, nil, err
	}
	return job, summary, nil
}

func decodeSummary(data []byte) (*protocol.Summary, error) {
	var summary protocol.Summary
	if err := json.Unmarshal(data, &summary); err != nil {
		return nil, protocol.NewServiceError(protocol.ClassStorage, "STATE_DECODE_FAILED",
			"could not decode stored statistical state", err)
	}
	return &summary, nil
}

func mustJSON(value any) []byte {
	data, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("internal JSON encoding failure: %v", err))
	}
	return data
}

func mapReadError(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return protocol.NewServiceError(protocol.ClassNotFound, "NOT_FOUND", "job or shard was not found", err)
	}
	if isResourceError(err) {
		return protocol.NewServiceError(protocol.ClassResourceExhausted, "DATABASE_RESOURCE_EXHAUSTED",
			"PostgreSQL rejected the request because a local resource limit was reached", err)
	}
	return protocol.NewServiceError(protocol.ClassStorage, "DATABASE_READ_FAILED", "database read failed", err)
}

func mapWriteError(err error) error {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		// 23505 = unique_violation. Callers map a duplicate shard explicitly
		// before insertion; a duplicate run_id is a state conflict.
		if pgErr.Code == "23505" {
			return protocol.NewServiceError(protocol.ClassState, "UNIQUE_CONSTRAINT_VIOLATION",
				"a job, run, or shard with that identifier already exists", err)
		}
	}
	if isResourceError(err) {
		return protocol.NewServiceError(protocol.ClassResourceExhausted, "DATABASE_RESOURCE_EXHAUSTED",
			"PostgreSQL rejected the write because a local resource limit was reached", err)
	}
	return protocol.NewServiceError(protocol.ClassStorage, "DATABASE_WRITE_FAILED", "database write failed", err)
}

func isResourceError(err error) bool {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		switch pgErr.Code {
		case "53000", "53100", "53200", "53300", "53400", "54000", "54100", "57P03":
			return true
		}
	}
	return false
}
