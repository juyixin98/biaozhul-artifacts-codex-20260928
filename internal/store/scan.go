package store

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5"

	"opp300-speculative-batch/internal/protocol"
)

const jobColumns = `id, idempotency_key, state, version, total_shards,
  max_attempts_per_shard, max_active_attempts, lease_ms, speculate_after_ms,
  heartbeat_interval_ms, failure_category, failure_message, created_at, updated_at`

const shardColumns = `id, job_id, ordinal, shard_key, input, behavior_json,
  state, attempt_number, attempt_count, failure_category, failure_message,
  created_at, updated_at`

const attemptColumns = `id, job_id, shard_id, attempt_number, kind, state, epoch,
  worker_url, reason, failure_category, failure_message, artifact_id,
  checksum_sha256, size_bytes, lease_expires_at, started_at, heartbeat_at,
  finished_at, created_at, updated_at`

type rowScanner interface {
	Scan(dest ...any) error
}

type querier interface {
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

func scanJob(row pgx.Row) (protocol.Job, error) {
	var job protocol.Job
	var holder policyHolder
	err := row.Scan(
		&job.ID, &job.IDempotencyKey, &job.State, &job.Version, &job.TotalShards,
		&holder.maxAttempts, &holder.maxActive, &holder.leaseMS, &holder.speculateMS,
		&holder.heartbeatMS, &job.FailureCategory, &job.FailureMessage, &job.CreatedAt, &job.UpdatedAt,
	)
	return job, err
}

func scanJobFull(row pgx.Row) (protocol.Job, policyHolder, error) {
	var job protocol.Job
	var holder policyHolder
	err := row.Scan(
		&job.ID, &job.IDempotencyKey, &job.State, &job.Version, &job.TotalShards,
		&holder.maxAttempts, &holder.maxActive, &holder.leaseMS, &holder.speculateMS,
		&holder.heartbeatMS, &job.FailureCategory, &job.FailureMessage, &job.CreatedAt, &job.UpdatedAt,
	)
	return job, holder, err
}

type policyHolder struct {
	maxAttempts int
	maxActive   int
	leaseMS     int
	speculateMS int
	heartbeatMS int
}

func policyFromJob(job protocol.Job) protocol.Policy {
	// Dispatch callers use scanJobWithPolicy because SELECT includes policy columns.
	return protocol.Policy{}.Normalized()
}

func policyFromHolder(_ protocol.Job, holder policyHolder) protocol.Policy {
	return protocol.Policy{
		MaxAttemptsPerShard: holder.maxAttempts,
		MaxActiveAttempts:   holder.maxActive,
		LeaseMS:             holder.leaseMS,
		SpeculateAfterMS:    holder.speculateMS,
		HeartbeatIntervalMS: holder.heartbeatMS,
	}.Normalized()
}

func scanShard(row pgx.Row) (protocol.Shard, error) {
	var shard protocol.Shard
	var behavior []byte
	err := row.Scan(
		&shard.ID, &shard.JobID, &shard.Ordinal, &shard.Key, &shard.Input, &behavior,
		&shard.State, &shard.AttemptNumber, &shard.AttemptCount,
		&shard.FailureCategory, &shard.FailureMessage, &shard.CreatedAt, &shard.UpdatedAt,
	)
	if err == nil {
		err = json.Unmarshal(behavior, &shard.Behavior)
	}
	return shard, err
}

func scanAttempt(row pgx.Row) (protocol.Attempt, error) {
	var a protocol.Attempt
	err := row.Scan(
		&a.ID, &a.JobID, &a.ShardID, &a.AttemptNumber, &a.Kind, &a.State, &a.Epoch,
		&a.WorkerURL, &a.Reason, &a.FailureCategory, &a.FailureMessage, &a.ArtifactID,
		&a.ChecksumSHA256, &a.SizeBytes, &a.LeaseExpiresAt, &a.StartedAt,
		&a.HeartbeatAt, &a.FinishedAt, &a.CreatedAt, &a.UpdatedAt,
	)
	return a, err
}

func scanEvent(row pgx.Row) (protocol.Event, error) {
	var event protocol.Event
	var payload []byte
	err := row.Scan(&event.ID, &event.JobID, &event.ShardID, &event.AttemptID,
		&event.Type, &event.Version, &event.Epoch, &payload, &event.CreatedAt)
	if err == nil && len(payload) > 0 {
		err = json.Unmarshal(payload, &event.Payload)
	}
	return event, err
}

func collectShards(rows pgx.Rows) ([]protocol.Shard, error) {
	defer rows.Close()
	var out []protocol.Shard
	for rows.Next() {
		shard, err := scanShard(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, shard)
	}
	return out, rows.Err()
}

func collectAttempts(rows pgx.Rows) ([]protocol.Attempt, error) {
	defer rows.Close()
	var out []protocol.Attempt
	for rows.Next() {
		attempt, err := scanAttempt(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, attempt)
	}
	return out, rows.Err()
}

func queryShards(ctx context.Context, q querier, jobID string) ([]protocol.Shard, error) {
	rows, err := q.Query(ctx, `SELECT `+shardColumns+` FROM shards WHERE job_id=$1 ORDER BY ordinal`, jobID)
	if err != nil {
		return nil, err
	}
	return collectShards(rows)
}
