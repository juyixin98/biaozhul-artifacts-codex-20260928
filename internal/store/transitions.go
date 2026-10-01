package store

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"opp300-speculative-batch/internal/artifacts"
	"opp300-speculative-batch/internal/protocol"
)

type Publisher interface {
	Commit(input artifacts.CommitInput) (artifacts.CommittedArtifact, error)
}

type PublishedReader interface {
	ReadPublished(rel string) ([]byte, error)
}

var ErrSuperseded = errors.New("attempt superseded")

type RunningInput struct {
	Now       time.Time
	Epoch     string
	LeaseFor  time.Duration
	WorkerURL string
}

type AttemptFailure struct {
	Now       time.Time
	Epoch     string
	Category  string
	Message   string
	WorkerURL string
}

type CompleteInput struct {
	AttemptID string
	Epoch     string
	Now       time.Time
	Staged    artifacts.StagedArtifact
	Publisher Publisher
}

type CompleteResult struct {
	Attempt protocol.Attempt
	Shard   protocol.Shard
	Job     protocol.Job
}

func isActive(state string) bool {
	return state == protocol.AttemptPending || state == protocol.AttemptRunning
}

func normalizeFailureCategory(category string) string {
	switch strings.ToUpper(strings.TrimSpace(category)) {
	case protocol.FailureValidation:
		return protocol.FailureValidation
	case protocol.FailureWorker:
		return protocol.FailureWorker
	case protocol.FailureLeaseExpired:
		return protocol.FailureLeaseExpired
	case protocol.FailureInterrupted:
		return protocol.FailureInterrupted
	case protocol.FailureSuperseded:
		return protocol.FailureSuperseded
	default:
		return protocol.FailureCompute
	}
}

func validFailureCategory(category string) bool {
	switch strings.ToUpper(strings.TrimSpace(category)) {
	case protocol.FailureValidation, protocol.FailureCompute, protocol.FailureWorker,
		protocol.FailureLeaseExpired, protocol.FailureInterrupted:
		return true
	default:
		return false
	}
}

func lockAttempt(ctx context.Context, tx pgx.Tx, attemptID string) (protocol.Attempt, error) {
	return scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1 FOR UPDATE`, attemptID))
}

func lockShard(ctx context.Context, tx pgx.Tx, shardID string) (protocol.Shard, error) {
	return scanShard(tx.QueryRow(ctx, `SELECT `+shardColumns+` FROM shards WHERE id=$1 FOR UPDATE`, shardID))
}

func getJob(ctx context.Context, tx pgx.Tx, jobID string) (protocol.Job, error) {
	return scanJob(tx.QueryRow(ctx, `SELECT `+jobColumns+` FROM jobs WHERE id=$1 FOR UPDATE`, jobID))
}

func finishAttempt(ctx context.Context, tx pgx.Tx, now time.Time, attempt protocol.Attempt, state, category, message, workerURL string) error {
	if workerURL != "" {
		_, err := tx.Exec(ctx, `
UPDATE attempts
SET state=$2, failure_category=$3, failure_message=$4, worker_url=COALESCE(NULLIF($6,''), worker_url),
    finished_at=$5, lease_expires_at=$5, updated_at=$5
WHERE id=$1`,
			attempt.ID, state, category, message, now, workerURL)
		return err
	}
	_, err := tx.Exec(ctx, `
UPDATE attempts
SET state=$2, failure_category=$3, failure_message=$4,
    finished_at=$5, lease_expires_at=$5, updated_at=$5
WHERE id=$1`,
		attempt.ID, state, category, message, now)
	return err
}

func resetShardPending(ctx context.Context, tx pgx.Tx, now time.Time, shard protocol.Shard, attemptID, reason string) error {
	_, err := tx.Exec(ctx, `
UPDATE shards
SET state='pending', failure_category='', failure_message='', updated_at=$2
WHERE id=$1`, shard.ID, now)
	if err != nil {
		return err
	}
	return insertEvent(ctx, tx, shard.JobID, shard.ID, attemptID, "SHARD_RETRY", 0, "", map[string]any{
		"reason":        reason,
		"attempt_count": shard.AttemptCount,
		"shard_ordinal": shard.Ordinal,
		"next_attempt":  shard.AttemptCount + 1,
		"budget_basis":  "one original plus bounded speculative/retried attempts",
	})
}

func bumpJobVersion(ctx context.Context, tx pgx.Tx, jobID string, now time.Time) (protocol.Job, error) {
	return scanJob(tx.QueryRow(ctx, `
UPDATE jobs SET version=version+1, updated_at=$2 WHERE id=$1
RETURNING `+jobColumns, jobID, now))
}

func failShardAndMaybeJob(ctx context.Context, tx pgx.Tx, now time.Time, shard protocol.Shard, category, message, epoch string) error {
	_, err := tx.Exec(ctx, `
UPDATE shards
SET state='failed', failure_category=$2, failure_message=$3, updated_at=$4
WHERE id=$1`,
		shard.ID, category, message, now)
	if err != nil {
		return err
	}
	job, err := bumpJobVersion(ctx, tx, shard.JobID, now)
	if err != nil {
		return err
	}
	if err := insertEvent(ctx, tx, job.ID, shard.ID, "", "SHARD_FAILED", job.Version, epoch, map[string]any{
		"failure_category": category,
		"failure_message":  message,
		"shard_ordinal":    shard.Ordinal,
		"attempt_count":    shard.AttemptCount,
	}); err != nil {
		return err
	}

	var active, pending int
	if err := tx.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE state='running'), count(*) FILTER (WHERE state='pending')
FROM shards WHERE job_id=$1`, job.ID).Scan(&active, &pending); err != nil {
		return err
	}
	if active == 0 && pending == 0 {
		_, err = tx.Exec(ctx, `
UPDATE jobs
SET state='failed', failure_category=$2, failure_message=$3, version=version+1, updated_at=$4
WHERE id=$1 AND state IN ('pending','running')`,
			job.ID, category, message, now)
		if err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `
UPDATE attempts
SET state='superseded', failure_category='SUPERSEDED',
    failure_message='job terminated after another shard failed',
    finished_at=$2, lease_expires_at=$2, updated_at=$2
WHERE job_id=$1 AND state IN ('pending','running')`, job.ID, now)
		if err != nil {
			return err
		}
		if err := insertEvent(ctx, tx, job.ID, "", "", "JOB_FAILED", job.Version+1, epoch, map[string]any{
			"failure_category": category,
			"failure_message":  message,
			"basis":            "all logical shards reached terminal state and at least one failed",
		}); err != nil {
			return err
		}
	}
	return nil
}

func continueOrRetryOrFail(ctx context.Context, tx pgx.Tx, now time.Time, shard protocol.Shard, epoch, category, message string) error {
	otherActive, err := otherActiveCount(ctx, tx, shard.ID, "")
	if err != nil {
		return err
	}
	if otherActive > 0 {
		eventType := "SHARD_ATTEMPT_FAILED_WAITING_FOR_SPECULATION"
		if shard.AttemptCount >= shardAttemptBudget(ctx, tx, shard.JobID) {
			eventType = "SPECULATION_BUDGET"
		}
		return insertEvent(ctx, tx, shard.JobID, shard.ID, "", eventType, 0, epoch, map[string]any{
			"failure_category": category,
			"failure_message":  message,
			"other_active":     otherActive,
			"attempt_count":    shard.AttemptCount,
			"max_attempts":     shardAttemptBudget(ctx, tx, shard.JobID),
			"allowed":          shard.AttemptCount < shardAttemptBudget(ctx, tx, shard.JobID),
			"reason":           "waiting_for_existing_attempt",
		})
	}
	budget := shardAttemptBudget(ctx, tx, shard.JobID)
	if shard.AttemptCount < budget {
		return resetShardPending(ctx, tx, now, shard, "", "attempt_failed_retry")
	}
	return failShardAndMaybeJob(ctx, tx, now, shard, category, message, epoch)
}

func otherActiveCount(ctx context.Context, tx pgx.Tx, shardID, excludeAttemptID string) (int, error) {
	var count int
	err := tx.QueryRow(ctx, `
SELECT count(*) FROM attempts
WHERE shard_id=$1 AND ($2='' OR id<>$2)
  AND state IN ('pending','running')`, shardID, excludeAttemptID).Scan(&count)
	return count, err
}

func markJobSucceeded(ctx context.Context, tx pgx.Tx, jobID string, now time.Time, epoch string) (protocol.Job, error) {
	job, err := scanJob(tx.QueryRow(ctx, `
UPDATE jobs
SET state='succeeded', failure_category='', failure_message='', version=version+1, updated_at=$2
WHERE id=$1
RETURNING `+jobColumns, jobID, now))
	if err != nil {
		return protocol.Job{}, err
	}
	if err := insertEvent(ctx, tx, jobID, job.ID, "", "JOB_SUCCEEDED", job.Version, epoch, map[string]any{
		"total_shards": job.TotalShards,
		"basis":        "all logical shards published exactly one artifact",
	}); err != nil {
		return protocol.Job{}, err
	}
	return job, nil
}

func ensureEpoch(attempt protocol.Attempt, epoch string) error {
	if attempt.Epoch != epoch {
		return fmt.Errorf("%w: attempt belongs to %s, coordinator is %s", ErrFenced, attempt.Epoch, epoch)
	}
	return nil
}

func errInvalidAttemptState(attempt protocol.Attempt) error {
	return fmt.Errorf("%w: attempt %s is %s", ErrInvalidState, attempt.ID, attempt.State)
}
