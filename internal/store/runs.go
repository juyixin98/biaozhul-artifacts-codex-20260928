package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"
)

func InsertRun(ctx context.Context, q queryer, run Run) (int64, error) {
	var jobID any
	if run.JobID != nil {
		jobID = *run.JobID
	}
	var request any
	if len(run.Request) > 0 {
		request = []byte(run.Request)
	}
	var errPayload any
	if len(run.Error) > 0 {
		errPayload = []byte(run.Error)
	}
	row := q.QueryRowContext(ctx, `
INSERT INTO runs (id, job_id, route, method, status_code, status, request, error, started_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
RETURNING run_number`,
		run.ID, jobID, run.Route, run.Method, nullableInt(run.StatusCode),
		run.Status, request, errPayload, run.StartedAt)
	var n int64
	if err := row.Scan(&n); err != nil {
		return 0, mapDBError(err)
	}
	return n, nil
}

func (d *DB) InsertRun(ctx context.Context, run Run) (int64, error) {
	var n int64
	err := d.InTx(ctx, func(tx *sql.Tx) error {
		var err error
		n, err = InsertRun(ctx, tx, run)
		return err
	})
	return n, err
}

func (d *DB) CompleteRun(ctx context.Context, runID string, statusCode int, status string, errPayload json.RawMessage, finishedAt time.Time) error {
	var payload any
	if len(errPayload) > 0 {
		payload = []byte(errPayload)
	}
	_, err := d.db.ExecContext(ctx, `
UPDATE runs
SET status_code = $2, status = $3, error = $4, finished_at = $5
WHERE id = $1`, runID, statusCode, status, payload, finishedAt)
	return mapDBError(err)
}

func (d *DB) AttachRunJob(ctx context.Context, runID, jobID string) error {
	_, err := d.db.ExecContext(ctx,
		`UPDATE runs SET job_id = $2 WHERE id = $1 AND job_id IS NULL`,
		runID, jobID)
	return mapDBError(err)
}

func (d *DB) GetRun(ctx context.Context, runNumber int64) (Run, []RunEvent, error) {
	var run Run
	var jobID sql.NullString
	var statusCode sql.NullInt64
	var request sql.NullString
	var errString sql.NullString
	var finished sql.NullTime
	err := d.db.QueryRowContext(ctx, `
SELECT id, run_number, job_id, route, method, status_code, status,
       request, error, started_at, finished_at
FROM runs WHERE run_number = $1`, runNumber).Scan(
		&run.ID, &run.RunNumber, &jobID, &run.Route, &run.Method,
		&statusCode, &run.Status, &request, &errString,
		&run.StartedAt, &finished)
	if err != nil {
		return Run{}, nil, mapDBError(err)
	}
	if jobID.Valid {
		run.JobID = &jobID.String
	}
	if statusCode.Valid {
		run.StatusCode = int(statusCode.Int64)
	}
	if request.Valid {
		run.Request = json.RawMessage(request.String)
	}
	if errString.Valid {
		run.Error = json.RawMessage(errString.String)
	}
	if finished.Valid {
		t := finished.Time
		run.FinishedAt = &t
	}
	events, err := d.ListRunEvents(ctx, run.ID)
	if err != nil {
		return Run{}, nil, err
	}
	return run, events, nil
}

func (d *DB) ListRuns(ctx context.Context, jobID, excludeRunID string, limit int) ([]Run, error) {
	rows, err := d.db.QueryContext(ctx, `
SELECT id, run_number, job_id, route, method, status_code, status,
       request, error, started_at, finished_at
FROM runs WHERE job_id = $1 AND id <> $2
ORDER BY run_number DESC
LIMIT $3`, jobID, excludeRunID, limit)
	if err != nil {
		return nil, mapDBError(err)
	}
	defer rows.Close()

	var out []Run
	for rows.Next() {
		var run Run
		var job sql.NullString
		var statusCode sql.NullInt64
		var request sql.NullString
		var errString sql.NullString
		var finished sql.NullTime
		if err := rows.Scan(&run.ID, &run.RunNumber, &job, &run.Route,
			&run.Method, &statusCode, &run.Status, &request, &errString,
			&run.StartedAt, &finished); err != nil {
			return nil, mapDBError(err)
		}
		if job.Valid {
			run.JobID = &job.String
		}
		if statusCode.Valid {
			run.StatusCode = int(statusCode.Int64)
		}
		if request.Valid {
			run.Request = json.RawMessage(request.String)
		}
		if errString.Valid {
			run.Error = json.RawMessage(errString.String)
		}
		if finished.Valid {
			t := finished.Time
			run.FinishedAt = &t
		}
		out = append(out, run)
	}
	return out, rows.Err()
}

func (d *DB) ListRunEvents(ctx context.Context, runID string) ([]RunEvent, error) {
	rows, err := d.db.QueryContext(ctx, `
SELECT run_id, event_index, stage, result, reason, state, created_at
FROM run_events WHERE run_id = $1 ORDER BY event_index`, runID)
	if err != nil {
		return nil, mapDBError(err)
	}
	defer rows.Close()

	var out []RunEvent
	for rows.Next() {
		var e RunEvent
		if err := rows.Scan(&e.RunID, &e.Index, &e.Stage, &e.Result,
			&e.Reason, &e.State, &e.CreatedAt); err != nil {
			return nil, mapDBError(err)
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

func (d *DB) AddRunEvent(ctx context.Context, event RunEvent) error {
	_, err := d.db.ExecContext(ctx, `
INSERT INTO run_events (run_id, event_index, stage, result, reason, state, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7)`,
		event.RunID, event.Index, event.Stage, event.Result,
		event.Reason, []byte(event.State), event.CreatedAt)
	return mapDBError(err)
}

func NextRunEventIndex(ctx context.Context, q queryer, runID string) (int64, error) {
	var n int64
	err := q.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(event_index) + 1, 1) FROM run_events WHERE run_id = $1`,
		runID).Scan(&n)
	if err != nil {
		return 0, mapDBError(err)
	}
	return n, nil
}

func (d *DB) NextRunEventIndex(ctx context.Context, runID string) (int64, error) {
	return NextRunEventIndex(ctx, d.db, runID)
}

func nullableInt(v int) any {
	if v == 0 {
		return nil
	}
	return v
}
