package store

import (
	"context"
	"time"

	"opp300-speculative-batch/internal/protocol"
)

func (s *Store) MarkAttemptRunning(ctx context.Context, attemptID string, in RunningInput) (protocol.Attempt, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return protocol.Attempt{}, err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003002)`); err != nil {
		return protocol.Attempt{}, err
	}
	attempt, err := lockAttempt(ctx, tx, attemptID)
	if err != nil {
		return protocol.Attempt{}, mapNotFound(err)
	}
	if err := ensureEpoch(attempt, in.Epoch); err != nil {
		return protocol.Attempt{}, err
	}
	if !isActive(attempt.State) {
		return protocol.Attempt{}, errInvalidAttemptState(attempt)
	}
	lease := in.Now.Add(in.LeaseFor)
	_, err = tx.Exec(ctx, `
UPDATE attempts
SET state='running', worker_url=$2, started_at=COALESCE(started_at,$3),
    heartbeat_at=$3, lease_expires_at=$4, updated_at=$3
WHERE id=$1`, attemptID, in.WorkerURL, in.Now, lease)
	if err != nil {
		return protocol.Attempt{}, err
	}
	if err := insertEvent(ctx, tx, attempt.JobID, attempt.ShardID, attemptID, "ATTEMPT_RUNNING", 0, in.Epoch, map[string]any{
		"worker_url": in.WorkerURL, "lease_expires_at": lease,
	}); err != nil {
		return protocol.Attempt{}, err
	}
	attempt, err = scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attemptID))
	if err != nil {
		return protocol.Attempt{}, err
	}
	return attempt, tx.Commit(ctx)
}

func (s *Store) Heartbeat(ctx context.Context, attemptID, epoch string, now time.Time, leaseFor time.Duration) (protocol.Attempt, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return protocol.Attempt{}, err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003002)`); err != nil {
		return protocol.Attempt{}, err
	}
	attempt, err := lockAttempt(ctx, tx, attemptID)
	if err != nil {
		return protocol.Attempt{}, mapNotFound(err)
	}
	if err := ensureEpoch(attempt, epoch); err != nil {
		return protocol.Attempt{}, err
	}
	if attempt.State != protocol.AttemptRunning {
		return protocol.Attempt{}, errInvalidAttemptState(attempt)
	}
	lease := now.Add(leaseFor)
	_, err = tx.Exec(ctx, `
UPDATE attempts
SET heartbeat_at=$2, lease_expires_at=$3, updated_at=$2
WHERE id=$1`, attemptID, now, lease)
	if err != nil {
		return protocol.Attempt{}, err
	}
	if err := insertEvent(ctx, tx, attempt.JobID, attempt.ShardID, attemptID, "ATTEMPT_HEARTBEAT", 0, epoch, map[string]any{
		"heartbeat_at": now, "lease_expires_at": lease,
	}); err != nil {
		return protocol.Attempt{}, err
	}
	attempt, err = scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attemptID))
	if err != nil {
		return protocol.Attempt{}, err
	}
	return attempt, tx.Commit(ctx)
}

func (s *Store) FailAttempt(ctx context.Context, attemptID string, failure AttemptFailure) (protocol.Attempt, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return protocol.Attempt{}, err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003002)`); err != nil {
		return protocol.Attempt{}, err
	}
	attempt, err := scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attemptID))
	if err != nil {
		return protocol.Attempt{}, mapNotFound(err)
	}
	shard, err := lockShard(ctx, tx, attempt.ShardID)
	if err != nil {
		return protocol.Attempt{}, err
	}
	attempt, err = lockAttempt(ctx, tx, attemptID)
	if err != nil {
		return protocol.Attempt{}, err
	}
	if err := ensureEpoch(attempt, failure.Epoch); err != nil {
		return protocol.Attempt{}, err
	}
	if shard.State == protocol.ShardSucceeded || shard.State == protocol.ShardFailed {
		if isActive(attempt.State) {
			err = finishAttempt(ctx, tx, failure.Now, attempt, protocol.AttemptSuperseded,
				protocol.FailureSuperseded, "failure arrived after logical shard reached terminal state", failure.WorkerURL)
			if err != nil {
				return protocol.Attempt{}, err
			}
		}
		attempt, err = scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attemptID))
		if err != nil {
			return protocol.Attempt{}, err
		}
		return attempt, tx.Commit(ctx)
	}
	if !isActive(attempt.State) {
		return protocol.Attempt{}, errInvalidAttemptState(attempt)
	}

	category := normalizeFailureCategory(failure.Category)
	if err := finishAttempt(ctx, tx, failure.Now, attempt, protocol.AttemptFailed, category, failure.Message, failure.WorkerURL); err != nil {
		return protocol.Attempt{}, err
	}
	if err := insertEvent(ctx, tx, attempt.JobID, shard.ID, attemptID, "ATTEMPT_FAILED", 0, failure.Epoch, map[string]any{
		"failure_category": category,
		"failure_message":  failure.Message,
		"worker_url":       failure.WorkerURL,
		"attempt_number":   attempt.AttemptNumber,
		"kind":             attempt.Kind,
	}); err != nil {
		return protocol.Attempt{}, err
	}
	if err := continueOrRetryOrFail(ctx, tx, failure.Now, shard, failure.Epoch, category, failure.Message); err != nil {
		return protocol.Attempt{}, err
	}
	attempt, err = scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attemptID))
	if err != nil {
		return protocol.Attempt{}, err
	}
	return attempt, tx.Commit(ctx)
}
