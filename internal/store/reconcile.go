package store

import (
	"context"
	"time"

	"opp300-speculative-batch/internal/protocol"
)

func (s *Store) ReconcileExpired(ctx context.Context, now time.Time, epoch string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003001)`); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003002)`); err != nil {
		return err
	}
	rows, err := tx.Query(ctx, `
SELECT `+attemptColumns+` FROM attempts
WHERE state IN ('pending','running') AND lease_expires_at <= $1
ORDER BY created_at
FOR UPDATE SKIP LOCKED`, now)
	if err != nil {
		return err
	}
	expired, err := collectAttempts(rows)
	if err != nil {
		return err
	}
	for _, attempt := range expired {
		if err := finishAttempt(ctx, tx, now, attempt, protocol.AttemptExpired,
			protocol.FailureLeaseExpired, "worker did not renew the lease", ""); err != nil {
			return err
		}
		shard, err := lockShard(ctx, tx, attempt.ShardID)
		if err != nil {
			return err
		}
		if shard.State == protocol.ShardSucceeded || shard.State == protocol.ShardFailed {
			continue
		}
		if shard.AttemptCount < shardAttemptBudget(ctx, tx, shard.JobID) {
			if err := resetShardPending(ctx, tx, now, shard, attempt.ID, "lease_expired_retry"); err != nil {
				return err
			}
		} else {
			if err := failShardAndMaybeJob(ctx, tx, now, shard, protocol.FailureLeaseExpired,
				"attempt lease expired with no retry budget", epoch); err != nil {
				return err
			}
		}
	}
	return tx.Commit(ctx)
}

func (s *Store) RecoverInterrupted(ctx context.Context, now time.Time, newEpoch string) (int, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003001)`); err != nil {
		return 0, err
	}
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003002)`); err != nil {
		return 0, err
	}
	rows, err := tx.Query(ctx, `
SELECT `+attemptColumns+` FROM attempts
WHERE epoch<>$1 AND state IN ('pending','running')
ORDER BY created_at
FOR UPDATE SKIP LOCKED`, newEpoch)
	if err != nil {
		return 0, err
	}
	attempts, err := collectAttempts(rows)
	if err != nil {
		return 0, err
	}
	for _, attempt := range attempts {
		if err := finishAttempt(ctx, tx, now, attempt, protocol.AttemptInterrupted,
			protocol.FailureInterrupted, "coordinator restart fenced the attempt", ""); err != nil {
			return 0, err
		}
		shard, err := lockShard(ctx, tx, attempt.ShardID)
		if err != nil {
			return 0, err
		}
		if shard.State == protocol.ShardPending || shard.State == protocol.ShardRunning {
			if err := resetShardPending(ctx, tx, now, shard, attempt.ID, "coordinator_restart_retry"); err != nil {
				return 0, err
			}
		}
		if err := insertEvent(ctx, tx, attempt.JobID, shard.ID, attempt.ID, "COORDINATOR_EPOCH_FENCED", 0, newEpoch, map[string]any{
			"previous_epoch": attempt.Epoch,
			"new_epoch":      newEpoch,
			"attempt_state":  protocol.AttemptInterrupted,
		}); err != nil {
			return 0, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, err
	}
	return len(attempts), nil
}
