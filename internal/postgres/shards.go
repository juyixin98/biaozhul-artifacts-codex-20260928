package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"diststats/internal/protocol"
	"diststats/internal/stats"
)

func (s *Store) SubmitShard(ctx context.Context, jobID string, shard protocol.ShardRequest, incoming *protocol.Summary) (protocol.ShardResult, *protocol.Summary, *protocol.Summary, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, mapWriteError(err)
	}
	defer tx.Rollback(ctx)

	var job protocol.Job
	var beforeJSON []byte
	err = tx.QueryRow(ctx, `
SELECT id, run_id, dimension, estimator, expected_shards, status, received_shards, current_summary
FROM diststats_b.jobs WHERE id = $1 FOR UPDATE`, jobID).Scan(
		&job.ID, &job.RunID, &job.Dimension, &job.Estimator, &job.ExpectedShards, &job.Status,
		&job.ReceivedShards, &beforeJSON)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, mapReadError(err)
	}

	var existing string
	err = tx.QueryRow(ctx, `SELECT shard_id FROM diststats_b.shards WHERE job_id = $1 AND shard_id = $2`,
		jobID, shard.ShardID).Scan(&existing)
	if err == nil {
		return protocol.ShardResult{}, nil, nil, protocol.NewServiceError(protocol.ClassState, "DUPLICATE_SHARD",
			fmt.Sprintf("shard %q has already been merged into job %q", shard.ShardID, jobID), nil)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return protocol.ShardResult{}, nil, nil, mapReadError(err)
	}
	if job.Status == "complete" || (job.ExpectedShards > 0 && job.ReceivedShards >= job.ExpectedShards) {
		return protocol.ShardResult{}, nil, nil, protocol.NewServiceError(protocol.ClassState, "JOB_COMPLETE",
			fmt.Sprintf("job %q already has %d shards and accepts no more", jobID, job.ReceivedShards), nil)
	}

	before, err := decodeSummary(beforeJSON)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, err
	}
	after, err := stats.Merge(before, incoming)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, err
	}
	sequence := job.ReceivedShards + 1
	newStatus := "open"
	if job.ExpectedShards > 0 && sequence >= job.ExpectedShards {
		newStatus = "complete"
	}

	afterJSON, err := json.Marshal(after)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, encodingError("could not encode merged state", err)
	}
	payloadJSON, err := json.Marshal(shard)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, protocol.NewServiceError(protocol.ClassInput, "PAYLOAD_ENCODE_FAILED",
			"shard payload could not be retained for replay", err)
	}
	summaryJSON, err := json.Marshal(incoming)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, encodingError("could not encode shard summary", err)
	}

	var receivedAt time.Time
	err = tx.QueryRow(ctx, `
INSERT INTO diststats_b.shards (job_id, shard_id, sequence, producer_id, observations, weight, summary, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
RETURNING received_at`,
		jobID, shard.ShardID, sequence, shard.ProducerID, incoming.Count, incoming.Weight, summaryJSON, payloadJSON).
		Scan(&receivedAt)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, mapWriteError(err)
	}

	beforeView := before.View(job.Estimator)
	afterView := after.View(job.Estimator)
	_, err = tx.Exec(ctx, `
INSERT INTO diststats_b.events (job_id, sequence, run_id, shard_id, type, reason, before_state, after_state, payload)
VALUES ($1, $2, $3, $4, 'shard_accepted', 'unique shard passed dimension and numeric checks; merged Welford summary', $5, $6, $7)`,
		jobID, sequence, job.RunID, shard.ShardID, mustJSON(beforeView), mustJSON(afterView), payloadJSON)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, mapWriteError(err)
	}

	_, err = tx.Exec(ctx, `
UPDATE diststats_b.jobs
SET status = $2, received_shards = $3, current_summary = $4, updated_at = now()
WHERE id = $1`,
		jobID, newStatus, sequence, afterJSON)
	if err != nil {
		return protocol.ShardResult{}, nil, nil, mapWriteError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return protocol.ShardResult{}, nil, nil, mapWriteError(err)
	}

	return protocol.ShardResult{
		JobID:       jobID,
		ShardID:     shard.ShardID,
		Sequence:    int64(sequence),
		Observation: incoming.Count,
		Weight:      incoming.Weight,
		Summary:     incoming,
		ReceivedAt:  receivedAt,
	}, before, after, nil
}

func encodingError(message string, err error) error {
	return protocol.NewServiceError(protocol.ClassStorage, "STATE_ENCODE_FAILED", message, err)
}
