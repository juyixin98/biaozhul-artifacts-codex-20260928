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

func (s *Store) Claim(ctx context.Context, workerID string, currentEpoch int64) (Claim, bool, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return Claim{}, false, err
	}
	defer func() { _ = tx.Rollback() }()

	var jobID, shardID, token, kind string
	var attemptNo int
	var epoch int64
	err = tx.QueryRowContext(ctx, `
UPDATE attempts SET status='running', worker_id=$1::text, last_heartbeat_at=now()
WHERE (job_id, shard_id, attempt_no) IN (
  SELECT a.job_id,a.shard_id,a.attempt_no
  FROM attempts a
  JOIN shards sh ON sh.job_id=a.job_id AND sh.id=a.shard_id
  JOIN jobs j ON j.id=a.job_id
  WHERE a.status='ready' AND sh.status='running' AND j.status IN ('queued','running')
  ORDER BY a.started_at,a.job_id,a.shard_id,a.attempt_no
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
RETURNING job_id,shard_id,attempt_no,run_epoch,token,kind`, workerID).
		Scan(&jobID, &shardID, &attemptNo, &epoch, &token, &kind)
	if errors.Is(err, sql.ErrNoRows) {
		return Claim{}, false, nil
	}
	if err != nil {
		return Claim{}, false, err
	}
	var jobPayload, shardPayload []byte
	if err := tx.QueryRowContext(ctx, `SELECT payload FROM jobs WHERE id=$1`, jobID).Scan(&jobPayload); err != nil {
		return Claim{}, false, err
	}
	if err := tx.QueryRowContext(ctx, `SELECT payload FROM shards WHERE job_id=$1 AND id=$2`, jobID, shardID).Scan(&shardPayload); err != nil {
		return Claim{}, false, err
	}
	var manifest model.ManifestShard
	if err := json.Unmarshal(shardPayload, &manifest); err != nil {
		return Claim{}, false, err
	}
	var job model.Manifest
	if err := json.Unmarshal(jobPayload, &job); err != nil {
		return Claim{}, false, err
	}
	lease := model.Lease{
		JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, Token: token, Kind: kind,
		Records: manifest.Records, Behavior: manifest.Behavior, FailKinds: manifest.FailKinds,
		Kernel: kernel.Name, KernelVersion: kernel.Version, ProtocolVersion: "specbatch/v1",
		Extra: manifest.Extra,
	}
	if err := insertEventTx(ctx, tx, model.Event{
		JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, WorkerID: workerID,
		Type: "attempt_claimed", Status: model.AttemptRunning, Category: kind,
		Reason: fmt.Sprintf("worker=%s kernel=%s@%s", workerID, kernel.Name, kernel.Version),
		Payload: mustJSON(map[string]any{
			"input_records": len(manifest.Records), "record_ids": recordIDs(manifest.Records),
			"kernel": kernel.Name, "kernel_version": kernel.Version, "run_epoch": epoch,
		}),
	}); err != nil {
		return Claim{}, false, err
	}
	if err := tx.Commit(); err != nil {
		return Claim{}, false, err
	}
	return Claim{Epoch: epoch, Lease: lease}, true, nil
}

func (s *Store) Heartbeat(ctx context.Context, token, workerID string, progress float64) (int64, error) {
	if progress < 0 {
		progress = 0
	}
	if progress > 1 {
		progress = 1
	}
	var epoch int64
	var status string
	err := s.db.QueryRowContext(ctx, `
UPDATE attempts SET progress=$3,last_heartbeat_at=now()
WHERE token=$1 AND worker_id=$2
RETURNING run_epoch,status`, token, workerID, progress).Scan(&epoch, &status)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, ErrInvalidToken
	}
	if err != nil {
		return 0, err
	}
	if status != model.AttemptRunning {
		return epoch, ErrAttemptExpired
	}
	return epoch, nil
}

func (s *Store) ReportFailure(ctx context.Context, token, workerID, category, message string) error {
	if category == "" || category == "success" {
		category = model.FailureUnknown
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	var jobID, shardID string
	var attemptNo int
	var status, kind string
	err = tx.QueryRowContext(ctx, `
UPDATE attempts SET status='failed', failure_category=$3, ended_at=now()
WHERE token=$1 AND worker_id=$2
RETURNING job_id,shard_id,attempt_no,status,kind`, token, workerID, category).
		Scan(&jobID, &shardID, &attemptNo, &status, &kind)
	if errors.Is(err, sql.ErrNoRows) {
		tx.Rollback()
		// Distinguish stale tokens from unknown worker identity.
		if exists, _ := s.attemptExists(ctx, token); exists {
			return ErrAttemptExpired
		}
		return ErrInvalidToken
	}
	if err != nil {
		return err
	}
	_ = kind
	if err := insertEventTx(ctx, tx, model.Event{
		JobID: jobID, ShardID: shardID, AttemptNo: attemptNo, WorkerID: workerID,
		Type: "attempt_failed", Status: model.AttemptFailed, Category: category, Reason: message,
		Payload: mustJSON(map[string]any{"failure_category": category, "message": message}),
	}); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *Store) attemptExists(ctx context.Context, token string) (bool, error) {
	var exists bool
	err := s.db.QueryRowContext(ctx, `SELECT exists(SELECT 1 FROM attempts WHERE token=$1)`, token).Scan(&exists)
	return exists, err
}

func recordIDs(records []model.InputRecord) []string {
	out := make([]string, 0, len(records))
	for _, r := range records {
		out = append(out, r.ID)
	}
	return out
}
