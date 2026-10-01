package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"specbatch/internal/kernel"
	"specbatch/internal/model"
)

func (s *Store) SubmitSuccess(ctx context.Context, in SubmitInput) (SubmitResult, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return SubmitResult{}, err
	}
	defer func() { _ = tx.Rollback() }()

	var jobID, shardID, status, kind string
	var attemptNo int
	var epoch int64
	err = tx.QueryRowContext(ctx, `
SELECT job_id,shard_id,attempt_no,run_epoch,status,kind FROM attempts WHERE token=$1 FOR UPDATE`, in.Token).
		Scan(&jobID, &shardID, &attemptNo, &epoch, &status, &kind)
	if errors.Is(err, sql.ErrNoRows) {
		return SubmitResult{}, ErrInvalidToken
	}
	if err != nil {
		return SubmitResult{}, err
	}
	if in.WorkerID != "" {
		var claimedBy string
		if err := tx.QueryRowContext(ctx, `SELECT worker_id FROM attempts WHERE token=$1`, in.Token).Scan(&claimedBy); err != nil {
			return SubmitResult{}, err
		}
		if claimedBy != in.WorkerID {
			return SubmitResult{}, ErrInvalidToken
		}
	}
	if status != model.AttemptRunning {
		return SubmitResult{}, ErrAttemptExpired
	}
	if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, jobID+"|"+shardID); err != nil {
		return SubmitResult{}, err
	}
	lease, err := leaseTx(ctx, tx, jobID, shardID, attemptNo, in.Token, kind, epoch)
	if err != nil {
		return SubmitResult{}, err
	}
	records, contentHash, verr := kernel.Validate(in.Document, lease)
	if verr != nil {
		if err := rejectOrFailTx(ctx, tx, jobID, shardID, attemptNo, in.WorkerID, model.AttemptFailed, model.FailureValidation, verr.Error()); err != nil {
			return SubmitResult{}, err
		}
		if err := tx.Commit(); err != nil {
			return SubmitResult{}, err
		}
		return SubmitResult{}, fmt.Errorf("%w: %v", ErrValidation, verr)
	}

	// The unique primary key on (job_id, shard_id) is the arbitration point.
	_, err = tx.ExecContext(ctx, `
INSERT INTO artifacts(job_id,shard_id,attempt_no,content_hash,size,content)
VALUES ($1,$2,$3,$4,$5,$6)`,
		jobID, shardID, attemptNo, contentHash, len(in.Content), in.Content)
	if err != nil {
		if isUnique(err) {
			if err := rejectOrFailTx(ctx, tx, jobID, shardID, attemptNo, in.WorkerID, model.AttemptRejected,
				model.FailureSupersededSubmission, "canonical artifact already exists for shard"); err != nil {
				return SubmitResult{}, err
			}
			if err := tx.Commit(); err != nil {
				return SubmitResult{}, err
			}
			return SubmitResult{Won: false, JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, Epoch: epoch}, nil
		}
		return SubmitResult{}, err
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE attempts SET status='succeeded', progress=1, ended_at=now()
WHERE job_id=$1 AND shard_id=$2 AND attempt_no=$3`, jobID, shardID, attemptNo); err != nil {
		return SubmitResult{}, err
	}
	if err := markTerminalTx(ctx, tx, jobID, shardID, model.ShardSucceeded, kind, "artifact committed as canonical winner"); err != nil {
		return SubmitResult{}, err
	}
	// Loser attempts for the same shard are explicitly rejected. The next
	// scheduler tick does not need to clean them up heuristically.
	if _, err := tx.ExecContext(ctx, `
UPDATE attempts SET status='rejected', failure_category=$4, ended_at=now()
WHERE job_id=$1 AND shard_id=$2 AND attempt_no <> $3 AND status IN ('ready','running')`,
		jobID, shardID, attemptNo, model.FailureSupersededSubmission); err != nil {
		return SubmitResult{}, err
	}
	if err := insertEventTx(ctx, tx, model.Event{
		JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, WorkerID: in.WorkerID,
		Type: "artifact_committed", Status: model.AttemptSucceeded, Category: kind,
		Reason: fmt.Sprintf("winner=%d sha256=%s records=%d", attemptNo, contentHash, len(records)),
		Payload: mustJSON(map[string]any{
			"winner_attempt_no": attemptNo, "sha256": contentHash, "size": len(in.Content),
			"record_count": len(records), "record_ids": outputIDs(records),
		}),
	}); err != nil {
		return SubmitResult{}, err
	}
	if err := refreshJobStateTx(ctx, tx, jobID); err != nil {
		return SubmitResult{}, err
	}
	if err := tx.Commit(); err != nil {
		return SubmitResult{}, err
	}
	return SubmitResult{Won: true, JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, Epoch: epoch}, nil
}

func rejectOrFailTx(ctx context.Context, tx *sql.Tx, jobID, shardID string, attemptNo int, workerID, status, category, message string) error {
	if _, err := tx.ExecContext(ctx, `
UPDATE attempts SET status=$4, failure_category=$5, ended_at=now()
WHERE job_id=$1 AND shard_id=$2 AND attempt_no=$3`,
		jobID, shardID, attemptNo, status, category); err != nil {
		return err
	}
	return insertEventTx(ctx, tx, model.Event{
		JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, WorkerID: workerID,
		Type: "attempt_rejected", Status: status, Category: category, Reason: message,
		Payload: mustJSON(map[string]any{"failure_category": category, "message": message}),
	})
}

func leaseTx(ctx context.Context, tx *sql.Tx, jobID, shardID string, attemptNo int, token, kind string, epoch int64) (model.Lease, error) {
	var jobRaw, shardRaw []byte
	if err := tx.QueryRowContext(ctx, `SELECT payload FROM jobs WHERE id=$1`, jobID).Scan(&jobRaw); err != nil {
		return model.Lease{}, err
	}
	if err := tx.QueryRowContext(ctx, `SELECT payload FROM shards WHERE job_id=$1 AND id=$2`, jobID, shardID).Scan(&shardRaw); err != nil {
		return model.Lease{}, err
	}
	var job model.Manifest
	var sh model.ManifestShard
	if err := json.Unmarshal(jobRaw, &job); err != nil {
		return model.Lease{}, err
	}
	if err := json.Unmarshal(shardRaw, &sh); err != nil {
		return model.Lease{}, err
	}
	return model.Lease{
		JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, Token: token, Kind: kind,
		Records: sh.Records, Behavior: sh.Behavior, FailKinds: sh.FailKinds, Extra: sh.Extra,
		Kernel: kernel.Name, KernelVersion: kernel.Version, ProtocolVersion: "specbatch/v1",
	}, nil
}

func outputIDs(records []model.OutputRecord) []string {
	out := make([]string, 0, len(records))
	for _, r := range records {
		out = append(out, r.ID)
	}
	return out
}
