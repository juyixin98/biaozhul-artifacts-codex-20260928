package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"checkpointlab/internal/protocol"
)

type queryer interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
	ExecContext(context.Context, string, ...any) (sql.Result, error)
}

func (s *PostgresStore) StartCheckpoint(ctx context.Context, snap Snapshot) error {
	base := mustJSON(snap.Base)
	targets := mustJSON(snap.Targets)
	seen := mustJSON(snap.SeenBarriers)
	offsets := mustJSON(snap.BaseOffsets)
	_, err := s.db.ExecContext(ctx, `
INSERT INTO checkpoints
  (job_id, epoch, attempt_id, mode, status, base_state, inflights, targets, seen_barriers, base_offsets, version)
VALUES ($1,$2,$3,$4,'ACTIVE',$5,'{}'::jsonb,$6,$7,$8,1)`,
		snap.JobID, snap.Epoch, snap.AttemptID, string(snap.Mode),
		base, targets, seen, offsets)
	return mapSQLError(err)
}

func (s *PostgresStore) AppendInflight(ctx context.Context, jobID string, epoch int64, attemptID string, ref RecordRef) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer rollback(tx)
	snap, err := lockSnapshot(ctx, tx, jobID, epoch)
	if err != nil {
		return err
	}
	if snap.Status != StatusActive || snap.AttemptID != attemptID {
		return protocol.NewTypedError(protocol.CategoryCheckpointScope, "checkpoint attempt is not active", false)
	}
	_, err = tx.ExecContext(ctx, `
INSERT INTO checkpoint_records
  (job_id, checkpoint_epoch, stream, offset_position, record_epoch, key_name, value_delta)
VALUES ($1,$2,$3,$4,$5,$6,$7)
ON CONFLICT (job_id, checkpoint_epoch, stream, offset_position) DO NOTHING`,
		jobID, epoch, ref.Stream, ref.Offset, ref.Epoch, ref.Key, ref.Value)
	if err != nil {
		return mapSQLError(err)
	}
	return mapSQLError(tx.Commit())
}

func (s *PostgresStore) ObserveBarrier(ctx context.Context, jobID string, epoch int64, attemptID string, b protocol.Barrier) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer rollback(tx)
	snap, err := lockSnapshot(ctx, tx, jobID, epoch)
	if err != nil {
		return err
	}
	if snap.Status != StatusActive || snap.AttemptID != attemptID {
		return protocol.NewTypedError(protocol.CategoryCheckpointScope, "checkpoint attempt is not active", false)
	}
	if snap.Targets[b.Stream].Seen {
		return protocol.NewTypedError(protocol.CategoryBarrierOrder,
			fmt.Sprintf("duplicate barrier stream=%s epoch=%d", b.Stream, epoch), false)
	}
	snap.Targets[b.Stream] = BarrierTarget{Offset: b.Offset, Seen: true}
	snap.SeenBarriers[b.Stream] = true
	snap.Version++
	if err := updateSnapshotMeta(ctx, tx, snap); err != nil {
		return err
	}
	return mapSQLError(tx.Commit())
}

func (s *PostgresStore) SealCheckpoint(ctx context.Context, jobID string, epoch int64, attemptID string, base State, records []RecordRef) (Snapshot, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return Snapshot{}, err
	}
	defer rollback(tx)
	snap, err := lockSnapshot(ctx, tx, jobID, epoch)
	if err != nil {
		return Snapshot{}, err
	}
	if snap.Status != StatusActive || snap.AttemptID != attemptID {
		return Snapshot{}, protocol.NewTypedError(protocol.CategoryCheckpointScope, "checkpoint attempt is not active", false)
	}
	for _, stream := range []string{protocol.StreamA, protocol.StreamB} {
		if !snap.Targets[stream].Seen {
			return Snapshot{}, protocol.NewTypedError(protocol.CategoryCheckpointScope,
				fmt.Sprintf("barrier for %s has not arrived", stream), false)
		}
	}
	existing, err := loadRecords(ctx, tx, jobID, epoch)
	if err != nil {
		return Snapshot{}, err
	}
	if len(existing) != len(records) {
		return Snapshot{}, protocol.NewTypedError(protocol.CategoryUncertainStorage,
			fmt.Sprintf("seal requested %d records, durable snapshot has %d", len(records), len(existing)), true)
	}
	grouped := map[string][]RecordRef{}
	for _, ref := range existing {
		grouped[ref.Stream] = append(grouped[ref.Stream], ref)
	}
	snap.Base = cloneState(base)
	snap.Inflights = grouped
	snap.Status = StatusSealed
	snap.Version++
	if err := updateSnapshotSealed(ctx, tx, snap); err != nil {
		return Snapshot{}, err
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE outputs SET committed=true WHERE job_id=$1 AND epoch<=$2`,
		jobID, epoch); err != nil {
		return Snapshot{}, mapSQLError(err)
	}
	if err := tx.Commit(); err != nil {
		return Snapshot{}, mapSQLError(err)
	}
	return snap, nil
}

func (s *PostgresStore) GetCheckpoint(ctx context.Context, jobID string, epoch int64) (Snapshot, bool, error) {
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return Snapshot{}, false, err
	}
	defer rollback(tx)
	snap, err := loadSnapshot(ctx, tx, jobID, epoch)
	if errors.Is(err, sql.ErrNoRows) {
		return Snapshot{}, false, tx.Commit()
	}
	if err != nil {
		return Snapshot{}, false, err
	}
	return snap, true, tx.Commit()
}

func (s *PostgresStore) LatestSealed(ctx context.Context, jobID string) (Snapshot, bool, error) {
	var epoch int64
	err := s.db.QueryRowContext(ctx,
		`SELECT epoch FROM checkpoints WHERE job_id=$1 AND status='SEALED' ORDER BY epoch DESC LIMIT 1`,
		jobID).Scan(&epoch)
	if errors.Is(err, sql.ErrNoRows) {
		return Snapshot{}, false, nil
	}
	if err != nil {
		return Snapshot{}, false, err
	}
	return s.GetCheckpoint(ctx, jobID, epoch)
}

func (s *PostgresStore) ActiveCheckpoint(ctx context.Context, jobID string) (Snapshot, bool, error) {
	var epoch int64
	err := s.db.QueryRowContext(ctx,
		`SELECT epoch FROM checkpoints WHERE job_id=$1 AND status='ACTIVE' ORDER BY epoch DESC LIMIT 1`,
		jobID).Scan(&epoch)
	if errors.Is(err, sql.ErrNoRows) {
		return Snapshot{}, false, nil
	}
	if err != nil {
		return Snapshot{}, false, err
	}
	return s.GetCheckpoint(ctx, jobID, epoch)
}

func (s *PostgresStore) Restore(ctx context.Context, jobID string) (RestorePoint, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return RestorePoint{}, err
	}
	defer rollback(tx)
	var epoch sql.NullInt64
	err = tx.QueryRowContext(ctx,
		`SELECT MAX(epoch) FROM checkpoints WHERE job_id=$1 AND status='SEALED'`, jobID).Scan(&epoch)
	if err != nil {
		return RestorePoint{}, err
	}
	point := RestorePoint{ReplayStart: map[string]int64{protocol.StreamA: 0, protocol.StreamB: 0}}
	if epoch.Valid {
		snap, err := loadSnapshot(ctx, tx, jobID, epoch.Int64)
		if err != nil {
			return RestorePoint{}, err
		}
		if _, err := tx.ExecContext(ctx,
			`UPDATE checkpoints SET status='SUPERSEDED' WHERE job_id=$1 AND status='ACTIVE' AND epoch>=$2`,
			jobID, epoch.Int64); err != nil {
			return RestorePoint{}, err
		}
		if _, err := tx.ExecContext(ctx,
			`DELETE FROM outputs WHERE job_id=$1 AND epoch>$2`, jobID, epoch.Int64); err != nil {
			return RestorePoint{}, err
		}
		if _, err := tx.ExecContext(ctx,
			`UPDATE outputs SET committed=true WHERE job_id=$1 AND epoch<=$2`,
			jobID, epoch.Int64); err != nil {
			return RestorePoint{}, err
		}
		point.Snapshot = snap
		point.CommittedEpoch = snap.Epoch
		point.ReplayStart = map[string]int64{}
		for stream, target := range snap.Targets {
			point.ReplayStart[stream] = target.Offset
		}
	} else {
		if _, err := tx.ExecContext(ctx,
			`UPDATE checkpoints SET status='SUPERSEDED' WHERE job_id=$1 AND status='ACTIVE'`, jobID); err != nil {
			return RestorePoint{}, err
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM outputs WHERE job_id=$1`, jobID); err != nil {
			return RestorePoint{}, err
		}
	}
	if err := tx.Commit(); err != nil {
		return RestorePoint{}, mapSQLError(err)
	}
	return point, nil
}

func lockSnapshot(ctx context.Context, tx *sql.Tx, jobID string, epoch int64) (Snapshot, error) {
	row := tx.QueryRowContext(ctx, `SELECT * FROM checkpoints WHERE job_id=$1 AND epoch=$2 FOR UPDATE`, jobID, epoch)
	return scanSnapshot(row)
}

func loadSnapshot(ctx context.Context, q queryer, jobID string, epoch int64) (Snapshot, error) {
	row := q.QueryRowContext(ctx, `SELECT * FROM checkpoints WHERE job_id=$1 AND epoch=$2`, jobID, epoch)
	snap, err := scanSnapshot(row)
	if err != nil {
		return snap, err
	}
	refs, err := loadRecords(ctx, q, jobID, epoch)
	if err != nil {
		return snap, err
	}
	snap.Inflights = map[string][]RecordRef{}
	for _, ref := range refs {
		snap.Inflights[ref.Stream] = append(snap.Inflights[ref.Stream], ref)
	}
	return snap, nil
}

type scanner interface{ Scan(...any) error }

func scanSnapshot(row scanner) (Snapshot, error) {
	var snap Snapshot
	var mode, status string
	var base, inflights, targets, seen, offsets []byte
	var sealedAt sql.NullTime
	err := row.Scan(
		&snap.JobID, &snap.Epoch, &snap.AttemptID, &mode, &status,
		&base, &inflights, &targets, &seen, &offsets, &snap.Version,
		&snap.CreatedAt, &sealedAt,
	)
	if err != nil {
		return Snapshot{}, err
	}
	snap.Mode = protocol.Mode(mode)
	snap.Status = Status(status)
	if err := json.Unmarshal(base, &snap.Base); err != nil {
		return Snapshot{}, err
	}
	if err := json.Unmarshal(inflights, &snap.Inflights); err != nil {
		return Snapshot{}, err
	}
	if err := json.Unmarshal(targets, &snap.Targets); err != nil {
		return Snapshot{}, err
	}
	if err := json.Unmarshal(seen, &snap.SeenBarriers); err != nil {
		return Snapshot{}, err
	}
	if err := json.Unmarshal(offsets, &snap.BaseOffsets); err != nil {
		return Snapshot{}, err
	}
	return snap, nil
}

func loadRecords(ctx context.Context, q queryer, jobID string, epoch int64) ([]RecordRef, error) {
	rows, err := q.QueryContext(ctx, `
SELECT stream, offset_position, record_epoch, key_name, value_delta
FROM checkpoint_records
WHERE job_id=$1 AND checkpoint_epoch=$2
ORDER BY stream, offset_position`, jobID, epoch)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var refs []RecordRef
	for rows.Next() {
		var ref RecordRef
		if err := rows.Scan(&ref.Stream, &ref.Offset, &ref.Epoch, &ref.Key, &ref.Value); err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	return refs, rows.Err()
}

func updateSnapshotMeta(ctx context.Context, tx *sql.Tx, snap Snapshot) error {
	_, err := tx.ExecContext(ctx, `
UPDATE checkpoints SET targets=$1, seen_barriers=$2, version=$3
WHERE job_id=$4 AND epoch=$5`,
		mustJSON(snap.Targets), mustJSON(snap.SeenBarriers), snap.Version,
		snap.JobID, snap.Epoch)
	return mapSQLError(err)
}

func updateSnapshotSealed(ctx context.Context, tx *sql.Tx, snap Snapshot) error {
	_, err := tx.ExecContext(ctx, `
UPDATE checkpoints
SET status='SEALED', base_state=$1, inflights=$2, targets=$3, seen_barriers=$4,
    base_offsets=$5, version=$6, sealed_at=now()
WHERE job_id=$7 AND epoch=$8`,
		mustJSON(snap.Base), mustJSON(snap.Inflights), mustJSON(snap.Targets),
		mustJSON(snap.SeenBarriers), mustJSON(snap.BaseOffsets), snap.Version,
		snap.JobID, snap.Epoch)
	return mapSQLError(err)
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func mapSQLError(err error) error {
	if err == nil {
		return nil
	}
	var pqErr *pq.Error
	if errors.As(err, &pqErr) && pqErr.Code == "23505" {
		return protocol.NewTypedError(protocol.CategoryCheckpointScope, err.Error(), false)
	}
	return err
}
