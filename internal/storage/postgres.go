package storage

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	_ "github.com/lib/pq"

	"checkpointlab/internal/protocol"
)

type PostgresStore struct {
	db *sql.DB
}

func OpenPostgres(ctx context.Context, dsn string) (*PostgresStore, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, err
	}
	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	store := &PostgresStore{db: db}
	if err := store.Migrate(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func (s *PostgresStore) Migrate(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, Schema)
	return err
}

func (s *PostgresStore) Close() error { return s.db.Close() }

func (s *PostgresStore) CreateJob(ctx context.Context, jobID string, mode protocol.Mode, configJSON []byte) error {
	if len(configJSON) == 0 {
		configJSON = []byte("{}")
	}
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO jobs(id, mode, config) VALUES ($1,$2,$3)`,
		jobID, string(mode), string(configJSON))
	return mapSQLError(err)
}

func (s *PostgresStore) JobMode(ctx context.Context, jobID string) (protocol.Mode, error) {
	var mode string
	err := s.db.QueryRowContext(ctx, `SELECT mode FROM jobs WHERE id=$1`, jobID).Scan(&mode)
	if errors.Is(err, sql.ErrNoRows) {
		return "", fmt.Errorf("job %s not found", jobID)
	}
	return protocol.Mode(mode), err
}

func (s *PostgresStore) StageOutput(ctx context.Context, out Output) error {
	_, err := s.db.ExecContext(ctx, `
INSERT INTO outputs
  (output_id, job_id, stream, offset_position, epoch, key_name, value_delta, committed)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
ON CONFLICT (output_id) DO NOTHING`,
		out.OutputID, out.JobID, out.Stream, out.Offset, out.Epoch, out.Key, out.Value, out.Committed)
	return mapSQLError(err)
}

func (s *PostgresStore) ListOutputs(ctx context.Context, jobID string, committedOnly bool) ([]Output, error) {
	query := `SELECT output_id, job_id, stream, offset_position, epoch, key_name, value_delta, committed
	          FROM outputs WHERE job_id=$1`
	if committedOnly {
		query += ` AND committed=true`
	}
	query += ` ORDER BY epoch, stream, offset_position`
	rows, err := s.db.QueryContext(ctx, query, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var result []Output
	for rows.Next() {
		var out Output
		if err := rows.Scan(&out.OutputID, &out.JobID, &out.Stream, &out.Offset, &out.Epoch,
			&out.Key, &out.Value, &out.Committed); err != nil {
			return nil, err
		}
		result = append(result, out)
	}
	return result, rows.Err()
}

func (s *PostgresStore) AppendEvent(ctx context.Context, entry EventLogEntry) error {
	_, err := s.db.ExecContext(ctx, `
INSERT INTO request_events(job_id, request_id, epoch, step, location, detail)
VALUES ($1,$2,$3,$4,$5,$6)`,
		entry.JobID, entry.RequestID, entry.Epoch, entry.Step, entry.Location, entry.Detail)
	return err
}

func (s *PostgresStore) ListEvents(ctx context.Context, jobID string, limit int) ([]EventLogEntry, error) {
	query := `SELECT job_id, request_id, epoch, step, location, detail
	          FROM request_events WHERE job_id=$1 ORDER BY id`
	args := []any{jobID}
	if limit > 0 {
		query = `SELECT * FROM (
			SELECT job_id, request_id, epoch, step, location, detail
			FROM request_events WHERE job_id=$1 ORDER BY id DESC LIMIT $2
		) last ORDER BY epoch, step`
		args = append(args, limit)
	}
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var result []EventLogEntry
	for rows.Next() {
		var e EventLogEntry
		if err := rows.Scan(&e.JobID, &e.RequestID, &e.Epoch, &e.Step, &e.Location, &e.Detail); err != nil {
			return nil, err
		}
		result = append(result, e)
	}
	return result, rows.Err()
}

func rollback(tx *sql.Tx) {
	if tx != nil {
		_ = tx.Rollback()
	}
}
