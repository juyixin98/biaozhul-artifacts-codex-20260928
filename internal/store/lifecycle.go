package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"specbatch/internal/model"
)

func (s *Store) Bootstrap(ctx context.Context) (int64, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, schemaSQL); err != nil {
		return 0, err
	}
	var last sql.NullInt64
	if err := tx.QueryRowContext(ctx, `SELECT max(epoch) FROM coordinator_run`).Scan(&last); err != nil {
		return 0, err
	}
	res, err := tx.ExecContext(ctx, `
UPDATE attempts
SET status='failed', failure_category=$1::text, ended_at=now()
WHERE status IN ('ready','running')`, model.FailureCoordinatorRestart)
	if err != nil {
		return 0, err
	}
	inflight, _ := res.RowsAffected()
	epoch := last.Int64 + 1
	if _, err := tx.ExecContext(ctx, `INSERT INTO coordinator_run(epoch, started_at) VALUES ($1, now())`, epoch); err != nil {
		return 0, err
	}
	if err := insertEventTx(ctx, tx, model.Event{
		Type: "coordinator_started", Status: "running", Category: model.FailureCoordinatorRestart,
		Reason:  fmt.Sprintf("epoch=%d invalidated_inflight=%d", epoch, inflight),
		Payload: mustJSON(map[string]any{"epoch": epoch, "invalidated_inflight": inflight, "version": kernelVersion()}),
	}); err != nil {
		return 0, err
	}
	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return epoch, nil
}

func (s *Store) CreateJob(ctx context.Context, manifest model.Manifest) error {
	if manifest.SchemaVersion != "specbatch/manifest/v1" || manifest.JobID == "" || len(manifest.Shards) == 0 {
		return fmt.Errorf("%w: invalid manifest", ErrValidation)
	}
	if manifest.ProtocolVersion == "" {
		manifest.ProtocolVersion = "specbatch/v1"
	}
	if manifest.MaxAttemptsPerShard <= 0 || manifest.MaxSpeculativeCopies < 0 || manifest.SpecAfterMillis <= 0 {
		return fmt.Errorf("%w: invalid budget", ErrValidation)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	payload, _ := json.Marshal(manifest)
	_, err = tx.ExecContext(ctx, `
INSERT INTO jobs(id,status,total_shards,max_attempts_per_shard,max_speculative_copies,spec_after,payload)
VALUES ($1,'queued',$2,$3,$4, make_interval(secs => $5::double precision / 1000.0), $6)`,
		manifest.JobID, len(manifest.Shards), manifest.MaxAttemptsPerShard, manifest.MaxSpeculativeCopies,
		float64(manifest.SpecAfterMillis), payload)
	if err != nil {
		if isUnique(err) {
			return fmt.Errorf("%w: job already exists", ErrConflict)
		}
		return err
	}
	for i, sh := range manifest.Shards {
		if sh.ID == "" || len(sh.Records) == 0 {
			return fmt.Errorf("%w: invalid shard %s", ErrValidation, sh.ID)
		}
		shp, _ := json.Marshal(sh)
		if _, err := tx.ExecContext(ctx, `
INSERT INTO shards(job_id,id,idx,status,payload) VALUES ($1,$2,$3,'queued',$4)`,
			manifest.JobID, sh.ID, i, shp); err != nil {
			return err
		}
	}
	if err := insertEventTx(ctx, tx, model.Event{
		JobID: manifest.JobID, Type: "job_created", Status: model.JobQueued,
		Reason:  fmt.Sprintf("shards=%d max_speculative=%d spec_after_ms=%d", len(manifest.Shards), manifest.MaxSpeculativeCopies, manifest.SpecAfterMillis),
		Payload: payload,
	}); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *Store) GetJob(ctx context.Context, jobID string) (model.Job, error) {
	var j model.Job
	var millis int64
	err := s.db.QueryRowContext(ctx, `
SELECT id,status,total_shards,max_attempts_per_shard,max_speculative_copies,
       (EXTRACT(EPOCH FROM spec_after)*1000)::bigint,created_at,updated_at
FROM jobs WHERE id=$1`, jobID).Scan(
		&j.ID, &j.Status, &j.TotalShards, &j.MaxAttemptsPerShard, &j.MaxSpeculativeCopies,
		&millis, &j.CreatedAt, &j.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return j, ErrNotFound
	}
	j.SpecAfter = time.Duration(millis) * time.Millisecond
	return j, err
}

func (s *Store) GetManifest(ctx context.Context, jobID string) (model.Manifest, error) {
	var raw []byte
	err := s.db.QueryRowContext(ctx, `SELECT payload FROM jobs WHERE id=$1`, jobID).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return model.Manifest{}, ErrNotFound
	}
	if err != nil {
		return model.Manifest{}, err
	}
	var m model.Manifest
	return m, json.Unmarshal(raw, &m)
}

func insertEventTx(ctx context.Context, tx *sql.Tx, e model.Event) error {
	if e.Payload == nil {
		e.Payload = []byte(`{}`)
	}
	_, err := tx.ExecContext(ctx, `
INSERT INTO events(job_id,shard_id,attempt_no,worker_id,type,status,category,reason,payload)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
		e.JobID, e.ShardID, e.AttemptNo, e.WorkerID, e.Type, e.Status, e.Category, e.Reason, e.Payload)
	return err
}

func (s *Store) InsertEvent(ctx context.Context, e model.Event) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := insertEventTx(ctx, tx, e); err != nil {
		return err
	}
	return tx.Commit()
}
