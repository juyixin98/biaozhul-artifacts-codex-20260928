package store

import (
	"context"

	"opp300-speculative-batch/internal/artifacts"
	"opp300-speculative-batch/internal/protocol"
)

func (s *Store) CompleteAttempt(ctx context.Context, in CompleteInput) (CompleteResult, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return CompleteResult{}, err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(9003002)`); err != nil {
		return CompleteResult{}, err
	}
	attempt, err := scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, in.AttemptID))
	if err != nil {
		return CompleteResult{}, mapNotFound(err)
	}
	shard, err := lockShard(ctx, tx, attempt.ShardID)
	if err != nil {
		return CompleteResult{}, err
	}
	attempt, err = lockAttempt(ctx, tx, in.AttemptID)
	if err != nil {
		return CompleteResult{}, err
	}
	if err := ensureEpoch(attempt, in.Epoch); err != nil {
		return CompleteResult{}, err
	}

	if shard.State == protocol.ShardSucceeded {
		if isActive(attempt.State) {
			if err := finishAttempt(ctx, tx, in.Now, attempt, protocol.AttemptSuperseded,
				protocol.FailureSuperseded, "artifact arrived after another attempt had published", ""); err != nil {
				return CompleteResult{}, err
			}
		}
		if err := tx.Commit(ctx); err != nil {
			return CompleteResult{}, err
		}
		return CompleteResult{}, ErrSuperseded
	}
	if shard.State == protocol.ShardFailed || !isActive(attempt.State) {
		return CompleteResult{}, errInvalidAttemptState(attempt)
	}
	if in.Staged.ChecksumSHA256 == "" || in.Staged.SizeBytes < 0 || in.Publisher == nil {
		return CompleteResult{}, ErrChecksum
	}

	committed, err := in.Publisher.Commit(artifacts.CommitInput{Staged: in.Staged, JobID: attempt.JobID})
	if err != nil {
		return CompleteResult{}, err
	}
	_, err = tx.Exec(ctx, `
INSERT INTO artifacts (id, job_id, shard_id, attempt_id, storage_path, checksum_sha256, size_bytes, created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
		in.Staged.ID, attempt.JobID, shard.ID, attempt.ID, committed.StoragePath,
		in.Staged.ChecksumSHA256, in.Staged.SizeBytes, in.Now)
	if err != nil {
		return CompleteResult{}, err
	}
	_, err = tx.Exec(ctx, `
UPDATE attempts
SET state='succeeded', artifact_id=$2, checksum_sha256=$3, size_bytes=$4,
    finished_at=$5, lease_expires_at=$5, updated_at=$5
WHERE id=$1`,
		attempt.ID, in.Staged.ID, in.Staged.ChecksumSHA256, in.Staged.SizeBytes, in.Now)
	if err != nil {
		return CompleteResult{}, err
	}
	_, err = tx.Exec(ctx, `
UPDATE shards
SET state='succeeded', attempt_number=$2, failure_category='', failure_message='', updated_at=$3
WHERE id=$1`, shard.ID, attempt.AttemptNumber, in.Now)
	if err != nil {
		return CompleteResult{}, err
	}

	otherActive, err := otherActiveCount(ctx, tx, shard.ID, attempt.ID)
	if err != nil {
		return CompleteResult{}, err
	}
	if otherActive > 0 {
		_, err = tx.Exec(ctx, `
UPDATE attempts
SET state='superseded', failure_category='SUPERSEDED',
    failure_message='another attempt published the logical shard first',
    finished_at=$3, lease_expires_at=$3, updated_at=$3
WHERE shard_id=$1 AND id<>$2 AND state IN ('pending','running')`,
			shard.ID, attempt.ID, in.Now)
		if err != nil {
			return CompleteResult{}, err
		}
	}

	job, err := bumpJobVersion(ctx, tx, attempt.JobID, in.Now)
	if err != nil {
		return CompleteResult{}, err
	}
	if err := insertEvent(ctx, tx, job.ID, shard.ID, attempt.ID, "ARTIFACT_PUBLISHED", job.Version, in.Epoch, map[string]any{
		"winner_attempt_id":     attempt.ID,
		"winner_attempt_number": attempt.AttemptNumber,
		"winner_kind":           attempt.Kind,
		"checksum_sha256":       in.Staged.ChecksumSHA256,
		"size_bytes":            in.Staged.SizeBytes,
		"storage_path":          committed.StoragePath,
		"superseded_attempts":   otherActive,
		"basis":                 "shard row lock and state transition in the publish transaction",
	}); err != nil {
		return CompleteResult{}, err
	}

	var succeeded, terminal int
	if err := tx.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE state='succeeded'),
       count(*) FILTER (WHERE state IN ('succeeded','failed'))
FROM shards WHERE job_id=$1`, job.ID).Scan(&succeeded, &terminal); err != nil {
		return CompleteResult{}, err
	}
	if succeeded == job.TotalShards {
		job, err = markJobSucceeded(ctx, tx, job.ID, in.Now, in.Epoch)
		if err != nil {
			return CompleteResult{}, err
		}
	} else if err := insertEvent(ctx, tx, job.ID, "", "", "JOB_PROGRESS", job.Version, in.Epoch, map[string]any{
		"succeeded_shards": succeeded,
		"terminal_shards":  terminal,
		"total_shards":     job.TotalShards,
	}); err != nil {
		return CompleteResult{}, err
	}

	attempt, err = scanAttempt(tx.QueryRow(ctx, `SELECT `+attemptColumns+` FROM attempts WHERE id=$1`, attempt.ID))
	if err != nil {
		return CompleteResult{}, err
	}
	shard, err = lockShard(ctx, tx, shard.ID)
	if err != nil {
		return CompleteResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return CompleteResult{}, err
	}
	return CompleteResult{Attempt: attempt, Shard: shard, Job: job}, nil
}
