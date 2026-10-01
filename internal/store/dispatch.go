package store

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"opp300-speculative-batch/internal/idgen"
	"opp300-speculative-batch/internal/protocol"
)

type DispatchPlan struct {
	Started []protocol.Attempt
}

func (s *Store) PlanDispatch(ctx context.Context, now time.Time, epoch string) (DispatchPlan, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return DispatchPlan{}, err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003001)`); err != nil {
		return DispatchPlan{}, err
	}

	rows, err := tx.Query(ctx, `
SELECT `+jobColumns+` FROM jobs
WHERE state IN ('pending','running')
ORDER BY created_at
FOR UPDATE SKIP LOCKED`)
	if err != nil {
		return DispatchPlan{}, err
	}
	var loadedJobs []struct {
		job    protocol.Job
		policy protocol.Policy
	}
	for rows.Next() {
		job, holder, err := scanJobFull(rows)
		if err != nil {
			rows.Close()
			return DispatchPlan{}, err
		}
		loadedJobs = append(loadedJobs, struct {
			job    protocol.Job
			policy protocol.Policy
		}{job: job, policy: policyFromHolder(job, holder)})
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return DispatchPlan{}, err
	}

	plan := DispatchPlan{}
	for _, loaded := range loadedJobs {
		job, policy := loaded.job, loaded.policy
		active, err := countActiveAttempts(ctx, tx, job.ID)
		if err != nil {
			return DispatchPlan{}, err
		}

		pendingRows, err := tx.Query(ctx, `
SELECT `+shardColumns+` FROM shards
WHERE job_id=$1 AND state='pending'
ORDER BY ordinal
FOR UPDATE SKIP LOCKED`, job.ID)
		if err != nil {
			return DispatchPlan{}, err
		}
		pending, err := collectShards(pendingRows)
		if err != nil {
			return DispatchPlan{}, err
		}

		for len(pending) > 0 && active < policy.MaxActiveAttempts {
			shard := pending[0]
			pending = pending[1:]
			attempt, err := startAttempt(ctx, tx, now, epoch, job, policy, shard, protocol.AttemptPending, "initial_pending_shard")
			if err != nil {
				return DispatchPlan{}, err
			}
			plan.Started = append(plan.Started, attempt)
			active++
		}
		if len(pending) > 0 {
			if err := insertEvent(ctx, tx, job.ID, "", "", "SPECULATION_BUDGET", job.Version, epoch, map[string]any{
				"allowed": false, "reason": "global_active_attempt_capacity",
				"active_attempts": active, "max_active_attempts": policy.MaxActiveAttempts,
				"blocked_pending": len(pending),
			}); err != nil {
				return DispatchPlan{}, err
			}
		}

		candidateRows, err := tx.Query(ctx, `
SELECT `+shardColumns+` FROM shards s
WHERE s.job_id=$1 AND s.state='running'
  AND s.attempt_count < $2
  AND EXISTS (
    SELECT 1 FROM attempts a
    WHERE a.shard_id=s.id AND a.state IN ('pending','running')
    GROUP BY a.shard_id
    HAVING count(*) = 1 AND min(a.created_at) <= $3
  )
ORDER BY s.ordinal
FOR UPDATE OF s SKIP LOCKED`,
			job.ID, policy.MaxAttemptsPerShard, now.Add(-time.Duration(policy.SpeculateAfterMS)*time.Millisecond))
		if err != nil {
			return DispatchPlan{}, err
		}
		candidates, err := collectShards(candidateRows)
		if err != nil {
			return DispatchPlan{}, err
		}

		for _, shard := range candidates {
			if active >= policy.MaxActiveAttempts {
				if err := insertEvent(ctx, tx, job.ID, shard.ID, "", "SPECULATION_BUDGET", job.Version, epoch, map[string]any{
					"allowed": false, "reason": "global_active_attempt_capacity",
					"active_attempts": active, "max_active_attempts": policy.MaxActiveAttempts,
					"shard_ordinal": shard.Ordinal,
				}); err != nil {
					return DispatchPlan{}, err
				}
				continue
			}
			oldest, err := oldestActiveAttempt(ctx, tx, shard.ID)
			if err != nil {
				return DispatchPlan{}, err
			}
			attempt, err := startAttempt(ctx, tx, now, epoch, job, policy, shard, protocol.AttemptPending,
				fmt.Sprintf("older_than_%dms_age_ms_%d", policy.SpeculateAfterMS, now.Sub(oldest.CreatedAt).Milliseconds()))
			if err != nil {
				return DispatchPlan{}, err
			}
			plan.Started = append(plan.Started, attempt)
			active++
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return DispatchPlan{}, err
	}
	return plan, nil
}

func startAttempt(ctx context.Context, tx pgx.Tx, now time.Time, epoch string, job protocol.Job, policy protocol.Policy, shard protocol.Shard, state string, reason string) (protocol.Attempt, error) {
	row := tx.QueryRow(ctx, `
UPDATE jobs
SET state=CASE WHEN state='pending' THEN 'running' ELSE state END,
    version=version+1,
    updated_at=$2
WHERE id=$1
RETURNING `+jobColumns, job.ID, now)
	updatedJob, holder, err := scanJobFull(row)
	if err != nil {
		return protocol.Attempt{}, err
	}
	policy = policyFromHolder(updatedJob, holder)

	locked, err := lockShard(ctx, tx, shard.ID)
	if err != nil {
		return protocol.Attempt{}, err
	}
	if locked.State == protocol.ShardSucceeded || locked.State == protocol.ShardFailed {
		return protocol.Attempt{}, fmt.Errorf("%w: shard %s is terminal", ErrInvalidState, locked.ID)
	}
	nextNumber := locked.AttemptCount + 1
	kind := "original"
	var previousState string
	if nextNumber > 1 {
		err = tx.QueryRow(ctx, `
SELECT state FROM attempts
WHERE shard_id=$1
ORDER BY attempt_number DESC
LIMIT 1`, locked.ID).Scan(&previousState)
		if err != nil {
			return protocol.Attempt{}, err
		}
	}
	if nextNumber > 1 && previousState != protocol.AttemptInterrupted {
		kind = "speculative"
	}
	if previousState == protocol.AttemptInterrupted {
		reason = "coordinator_restart_replacement"
	}
	attemptID, err := idgen.New("att")
	if err != nil {
		return protocol.Attempt{}, err
	}
	leaseExpires := now.Add(time.Duration(policy.LeaseMS) * time.Millisecond)
	_, err = tx.Exec(ctx, `
INSERT INTO attempts (id, job_id, shard_id, attempt_number, kind, state, epoch, reason, lease_expires_at, created_at, updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$10)`,
		attemptID, job.ID, locked.ID, nextNumber, kind, state, epoch, reason, leaseExpires, now)
	if err != nil {
		return protocol.Attempt{}, err
	}
	_, err = tx.Exec(ctx, `
UPDATE shards
SET state='running', attempt_number=$2, attempt_count=attempt_count+1, updated_at=$3,
    failure_category='', failure_message=''
WHERE id=$1`,
		locked.ID, nextNumber, now)
	if err != nil {
		return protocol.Attempt{}, err
	}
	if err := insertEvent(ctx, tx, job.ID, locked.ID, attemptID, "SHARD_ATTEMPT_STARTED", updatedJob.Version, epoch, map[string]any{
		"shard_ordinal":       locked.Ordinal,
		"shard_key":           locked.Key,
		"attempt_number":      nextNumber,
		"kind":                kind,
		"reason":              reason,
		"lease_expires_at":    leaseExpires,
		"speculate_after_ms":  policy.SpeculateAfterMS,
		"max_active_attempts": policy.MaxActiveAttempts,
	}); err != nil {
		return protocol.Attempt{}, err
	}
	return scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attemptID))
}

func countActiveAttempts(ctx context.Context, tx pgx.Tx, jobID string) (int, error) {
	var active int
	err := tx.QueryRow(ctx, `
SELECT count(*) FROM attempts
WHERE job_id=$1 AND state IN ('pending','running')`, jobID).Scan(&active)
	return active, err
}

func oldestActiveAttempt(ctx context.Context, tx pgx.Tx, shardID string) (protocol.Attempt, error) {
	return scanAttempt(tx.QueryRow(ctx, `
SELECT `+attemptColumns+` FROM attempts
WHERE shard_id=$1 AND state IN ('pending','running')
ORDER BY created_at
LIMIT 1
FOR UPDATE`, shardID))
}

func shardAttemptBudget(ctx context.Context, tx pgx.Tx, jobID string) int {
	var value int
	if err := tx.QueryRow(ctx, `SELECT max_attempts_per_shard FROM jobs WHERE id=$1`, jobID).Scan(&value); err != nil {
		return 2
	}
	return value
}
