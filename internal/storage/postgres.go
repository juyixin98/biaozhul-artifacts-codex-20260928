package storage

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
)

type PostgresStore struct {
	pool          *pgxpool.Pool
	migrationPath string
}

func NewPostgresStore(dsn, migrationPath string) (*PostgresStore, error) {
	if migrationPath == "" {
		migrationPath = filepath.Join("migrations", "001_checkpoints.sql")
	}
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse postgres dsn: %w", err)
	}
	pool, err := pgxpool.NewWithConfig(context.Background(), cfg)
	if err != nil {
		return nil, fmt.Errorf("connect postgres: %w", err)
	}
	return &PostgresStore{pool: pool, migrationPath: migrationPath}, nil
}

func (s *PostgresStore) Close() {
	s.pool.Close()
}

func (s *PostgresStore) Ping(ctx context.Context) error {
	return s.pool.Ping(ctx)
}

func (s *PostgresStore) Init(ctx context.Context) error {
	sqlText, err := os.ReadFile(s.migrationPath)
	if err != nil {
		return fmt.Errorf("read migration: %w", err)
	}
	if _, err := s.pool.Exec(ctx, string(sqlText)); err != nil {
		return fmt.Errorf("apply migration: %w", err)
	}
	return nil
}

func (s *PostgresStore) SaveCheckpoint(ctx context.Context, cp protocol.Checkpoint) error {
	if cp.Status != protocol.CheckpointComplete {
		return &kernel.ClassifiedError{Category: kernel.CategoryStorage, Err: fmt.Errorf("only complete checkpoints are durable")}
	}
	stateJSON, err := json.Marshal(cp.State)
	if err != nil {
		return err
	}
	inflightJSON, err := json.Marshal(cp.InFlight)
	if err != nil {
		return err
	}
	positionsJSON, err := json.Marshal(cp.Positions)
	if err != nil {
		return err
	}
	metricsJSON, err := json.Marshal(cp.Metrics)
	if err != nil {
		return err
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin checkpoint transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	var maxID *int64
	if err := tx.QueryRow(ctx, `SELECT max(id) FROM checkpoints WHERE task_id=$1`, cp.TaskID).Scan(&maxID); err != nil {
		return err
	}
	if maxID != nil && *maxID > cp.ID {
		return &kernel.ClassifiedError{Category: kernel.CategoryProtocol, Err: kernel.ErrBarrierOutOfOrder}
	}

	conflict := false
	tag, err := tx.Exec(ctx, `
INSERT INTO checkpoints
    (task_id, id, mode, status, state, inflight, positions, metrics, started_at, finished_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
ON CONFLICT (task_id, id) DO NOTHING`,
		cp.TaskID, cp.ID, string(cp.Mode), string(cp.Status), stateJSON, inflightJSON, positionsJSON,
		metricsJSON, cp.StartedAt, cp.FinishedAt)
	if err != nil {
		return mapPGError(err)
	}
	conflict = tag.RowsAffected() == 0
	for _, out := range cp.Outputs {
		payload, err := json.Marshal(out)
		if err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `
INSERT INTO checkpoint_outputs
    (task_id, checkpoint_id, output_id, stream, key, seq, epoch, payload)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
ON CONFLICT (task_id, output_id) DO NOTHING`,
			cp.TaskID, cp.ID, out.ID, out.Stream, out.Key, out.Seq, out.Epoch, payload)
		if err != nil {
			return mapPGError(err)
		}
	}
	_ = conflict
	return tx.Commit(ctx)
}

func (s *PostgresStore) GetLatest(ctx context.Context, taskID string) (protocol.Checkpoint, error) {
	return s.GetCheckpoint(ctx, taskID, 0)
}

func (s *PostgresStore) GetCheckpoint(ctx context.Context, taskID string, id int64) (protocol.Checkpoint, error) {
	query := `
SELECT task_id, id, mode, status, state, inflight, positions, metrics, started_at, finished_at
FROM checkpoints
WHERE task_id=$1 AND ($2=0 OR id=$2)
ORDER BY id DESC
LIMIT 1`
	row := s.pool.QueryRow(ctx, query, taskID, id)
	cp, err := scanCheckpoint(row)
	if err != nil {
		return protocol.Checkpoint{}, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT payload FROM checkpoint_outputs
WHERE task_id=$1 AND checkpoint_id=$2
ORDER BY epoch, stream, seq`, taskID, cp.ID)
	if err != nil {
		return protocol.Checkpoint{}, err
	}
	defer rows.Close()
	for rows.Next() {
		var payload []byte
		if err := rows.Scan(&payload); err != nil {
			return protocol.Checkpoint{}, err
		}
		var out protocol.Output
		if err := json.Unmarshal(payload, &out); err != nil {
			return protocol.Checkpoint{}, err
		}
		cp.Outputs = append(cp.Outputs, out)
	}
	return cp, rows.Err()
}

func (s *PostgresStore) ListCheckpoints(ctx context.Context, taskID string, limit int) ([]protocol.Checkpoint, error) {
	if limit <= 0 {
		limit = 20
	}
	rows, err := s.pool.Query(ctx, `
SELECT task_id, id, mode, status, state, inflight, positions, metrics, started_at, finished_at
FROM checkpoints
WHERE task_id=$1
ORDER BY id DESC
LIMIT $2`, taskID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []protocol.Checkpoint
	for rows.Next() {
		cp, err := scanCheckpoint(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, cp)
	}
	return out, rows.Err()
}

type rowScanner interface {
	Scan(dest ...any) error
}

func scanCheckpoint(row rowScanner) (protocol.Checkpoint, error) {
	var (
		cp           protocol.Checkpoint
		mode, status string
		stateJSON    []byte
		inflightJSON []byte
		positionJSON []byte
		metricsJSON  []byte
	)
	err := row.Scan(
		&cp.TaskID, &cp.ID, &mode, &status, &stateJSON, &inflightJSON, &positionJSON,
		&metricsJSON, &cp.StartedAt, &cp.FinishedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return protocol.Checkpoint{}, &kernel.ClassifiedError{Category: kernel.CategoryNotFound, Err: kernel.ErrCheckpointNotFound}
	}
	if err != nil {
		return protocol.Checkpoint{}, err
	}
	cp.Mode = protocol.Mode(mode)
	cp.Status = protocol.CheckpointStatus(status)
	cp.State = protocol.NewState()
	if err := json.Unmarshal(stateJSON, cp.State); err != nil {
		return protocol.Checkpoint{}, err
	}
	cp.InFlight = map[string][]protocol.InFlight{}
	if err := json.Unmarshal(inflightJSON, &cp.InFlight); err != nil {
		return protocol.Checkpoint{}, err
	}
	cp.Positions = map[string]protocol.Position{}
	if err := json.Unmarshal(positionJSON, &cp.Positions); err != nil {
		return protocol.Checkpoint{}, err
	}
	if err := json.Unmarshal(metricsJSON, &cp.Metrics); err != nil {
		return protocol.Checkpoint{}, err
	}
	return cp, nil
}

func mapPGError(err error) error {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" {
		return &kernel.ClassifiedError{Category: kernel.CategoryStorage, Err: fmt.Errorf("%w: %s", kernel.ErrDuplicateBarrier, pgErr.ConstraintName)}
	}
	if strings.Contains(err.Error(), "connection") {
		return &kernel.ClassifiedError{Category: kernel.CategoryStorage, Err: err}
	}
	return err
}
