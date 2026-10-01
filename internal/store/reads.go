package store

import (
	"context"

	"opp300-speculative-batch/internal/protocol"
)

func (s *Store) GetJob(ctx context.Context, jobID string) (protocol.Job, error) {
	return scanJob(s.pool.QueryRow(ctx, `SELECT `+jobColumns+` FROM jobs WHERE id=$1`, jobID))
}

func (s *Store) ListShards(ctx context.Context, jobID string) ([]protocol.Shard, error) {
	return queryShards(ctx, s.pool, jobID)
}

func (s *Store) GetAttempt(ctx context.Context, attemptID string) (protocol.Attempt, error) {
	attempt, err := scanAttempt(s.pool.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attemptID))
	return attempt, mapNotFound(err)
}

func (s *Store) GetShard(ctx context.Context, shardID string) (protocol.Shard, error) {
	shard, err := scanShard(s.pool.QueryRow(ctx, `SELECT `+shardColumns+` FROM shards WHERE id=$1`, shardID))
	return shard, mapNotFound(err)
}

func (s *Store) GetPolicy(ctx context.Context, jobID string) (protocol.Policy, error) {
	var holder policyHolder
	err := s.pool.QueryRow(ctx, `
SELECT max_attempts_per_shard, max_active_attempts, lease_ms,
       speculate_after_ms, heartbeat_interval_ms
FROM jobs WHERE id=$1`, jobID).Scan(
		&holder.maxAttempts, &holder.maxActive, &holder.leaseMS,
		&holder.speculateMS, &holder.heartbeatMS)
	if err != nil {
		return protocol.Policy{}, mapNotFound(err)
	}
	return policyFromHolder(protocol.Job{ID: jobID}, holder), nil
}

func (s *Store) ListAttempts(ctx context.Context, jobID string) ([]protocol.Attempt, error) {
	rows, err := s.pool.Query(ctx, `
SELECT `+attemptColumns+` FROM attempts
WHERE job_id=$1
ORDER BY shard_id, attempt_number`, jobID)
	if err != nil {
		return nil, err
	}
	return collectAttempts(rows)
}

func (s *Store) ListEvents(ctx context.Context, jobID string, afterID int64, limit int) ([]protocol.Event, error) {
	if limit <= 0 || limit > 1000 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
SELECT id, job_id, shard_id, attempt_id, event_type, version, epoch, payload, created_at
FROM events
WHERE job_id=$1 AND id>$2
ORDER BY id
LIMIT $3`, jobID, afterID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []protocol.Event
	for rows.Next() {
		event, err := scanEvent(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, event)
	}
	return out, rows.Err()
}

func (s *Store) Progress(ctx context.Context, jobID string) (protocol.Progress, error) {
	job, err := s.GetJob(ctx, jobID)
	if err != nil {
		return protocol.Progress{}, mapNotFound(err)
	}
	var p protocol.Progress
	err = s.pool.QueryRow(ctx, `
SELECT
  count(*) FILTER (WHERE state='succeeded'),
  count(*) FILTER (WHERE state='failed'),
  count(*) FILTER (WHERE state='pending'),
  count(*) FILTER (WHERE state='running')
FROM shards WHERE job_id=$1`, jobID).
		Scan(&p.Succeeded, &p.Failed, &p.Pending, &p.Running)
	if err != nil {
		return protocol.Progress{}, err
	}
	err = s.pool.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE state IN ('pending','running')), count(*)
FROM attempts WHERE job_id=$1`, jobID).Scan(&p.ActiveAttempts, &p.AttemptsCreated)
	if err != nil {
		return protocol.Progress{}, err
	}
	p.State = job.State
	p.Version = job.Version
	p.TotalShards = job.TotalShards
	return p, nil
}

func (s *Store) MergeResult(ctx context.Context, jobID string, reader PublishedReader) (protocol.MergeResult, error) {
	job, err := s.GetJob(ctx, jobID)
	if err != nil {
		return protocol.MergeResult{}, mapNotFound(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT s.ordinal, s.shard_key, a.id, a.attempt_number, a.kind, ar.id,
       ar.checksum_sha256, ar.size_bytes, ar.storage_path
FROM artifacts ar
JOIN shards s ON s.id=ar.shard_id
JOIN attempts a ON a.id=ar.attempt_id
WHERE ar.job_id=$1
ORDER BY s.ordinal`, jobID)
	if err != nil {
		return protocol.MergeResult{}, err
	}
	defer rows.Close()

	result := protocol.MergeResult{JobID: jobID, State: job.State, Version: job.Version}
	seen := map[int]bool{}
	for rows.Next() {
		var item protocol.PublishedShardResult
		var storagePath string
		if err := rows.Scan(&item.Ordinal, &item.Key, &item.AttemptID, &item.AttemptNumber,
			&item.Kind, &item.ArtifactID, &item.ChecksumSHA256, &item.SizeBytes, &storagePath); err != nil {
			return protocol.MergeResult{}, err
		}
		if seen[item.Ordinal] {
			return protocol.MergeResult{}, ErrInvalidState
		}
		seen[item.Ordinal] = true
		output, err := reader.ReadPublished(storagePath)
		if err != nil {
			return protocol.MergeResult{}, err
		}
		item.Output = string(output)
		result.Shards = append(result.Shards, item)
	}
	if err := rows.Err(); err != nil {
		return protocol.MergeResult{}, err
	}
	return result, nil
}
