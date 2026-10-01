package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"

	"diststats/internal/apperror"
)

func (d *DB) CreateJob(ctx context.Context, job Job) error {
	return InsertJob(ctx, d.db, job)
}

func InsertJob(ctx context.Context, q queryer, job Job) error {
	_, err := q.ExecContext(ctx, `
INSERT INTO jobs (id, dim, status, aggregate, version, created_at)
VALUES ($1, $2, $3, $4, $5, $6)`,
		job.ID, job.Dim, job.Status, []byte(job.Aggregate), job.Version, job.CreatedAt)
	return mapDBError(err)
}

func (d *DB) GetJob(ctx context.Context, jobID string) (Job, error) {
	return GetJob(ctx, d.db, jobID)
}

func GetJob(ctx context.Context, q queryer, jobID string) (Job, error) {
	var job Job
	var finalized sql.NullTime
	err := q.QueryRowContext(ctx, `
SELECT id, dim, status, aggregate, version, created_at, finalized_at
FROM jobs WHERE id = $1`, jobID).Scan(
		&job.ID, &job.Dim, &job.Status, &job.Aggregate,
		&job.Version, &job.CreatedAt, &finalized)
	if err != nil {
		return Job{}, mapDBError(err)
	}
	if finalized.Valid {
		t := finalized.Time
		job.FinalizedAt = &t
	}
	return job, nil
}

func GetJobForUpdate(ctx context.Context, q queryer, jobID string) (Job, error) {
	var job Job
	var finalized sql.NullTime
	err := q.QueryRowContext(ctx, `
SELECT id, dim, status, aggregate, version, created_at, finalized_at
FROM jobs WHERE id = $1 FOR UPDATE`, jobID).Scan(
		&job.ID, &job.Dim, &job.Status, &job.Aggregate,
		&job.Version, &job.CreatedAt, &finalized)
	if err != nil {
		return Job{}, mapDBError(err)
	}
	if finalized.Valid {
		t := finalized.Time
		job.FinalizedAt = &t
	}
	return job, nil
}

func (d *DB) FinalizeJob(ctx context.Context, jobID string, at time.Time) (Job, error) {
	var job Job
	var finalized sql.NullTime
	err := d.db.QueryRowContext(ctx, `
UPDATE jobs
SET status = 'finalized', finalized_at = $2
WHERE id = $1
RETURNING id, dim, status, aggregate, version, created_at, finalized_at`,
		jobID, at).Scan(
		&job.ID, &job.Dim, &job.Status, &job.Aggregate,
		&job.Version, &job.CreatedAt, &finalized)
	if err != nil {
		return Job{}, mapDBError(err)
	}
	if finalized.Valid {
		t := finalized.Time
		job.FinalizedAt = &t
	}
	return job, nil
}

func UpdateJobAggregate(ctx context.Context, q queryer, jobID string, aggregate json.RawMessage, newVersion int64) error {
	res, err := q.ExecContext(ctx, `
UPDATE jobs SET aggregate = $2, version = $3 WHERE id = $1 AND version = $4`,
		jobID, []byte(aggregate), newVersion, newVersion-1)
	if err != nil {
		return mapDBError(err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return mapDBError(err)
	}
	if n == 0 {
		return apperror.Conflict("JOB_VERSION_CONFLICT",
			"job aggregate version changed concurrently")
	}
	return nil
}

type queryer interface {
	ExecContext(context.Context, string, ...any) (sql.Result, error)
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
	QueryRowContext(context.Context, string, ...any) *sql.Row
}
