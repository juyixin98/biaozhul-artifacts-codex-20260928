package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"specbatch/internal/model"
)

func (s *Store) Schedule(ctx context.Context, epoch int64, now time.Time) ([]model.Attempt, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id FROM jobs WHERE status IN ('queued','running') ORDER BY created_at,id`)
	if err != nil {
		return nil, err
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return nil, err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rows.Close()

	var created []model.Attempt
	for _, jobID := range ids {
		manifest, err := s.GetManifest(ctx, jobID)
		if err != nil {
			return nil, err
		}
		for _, sh := range manifest.Shards {
			a, launched, reason, err := s.scheduleShard(ctx, epoch, manifest, sh, now)
			if err != nil {
				return nil, err
			}
			if launched {
				created = append(created, a)
				_ = s.InsertEvent(ctx, model.Event{
					JobID: a.JobID, ShardID: a.ShardID, AttemptNo: a.AttemptNo,
					Type: "attempt_launched", Status: a.Status, Category: a.Kind, Reason: reason,
					Payload: mustJSON(map[string]any{"attempt_no": a.AttemptNo, "kind": a.Kind, "job_budget": manifest.MaxSpeculativeCopies}),
				})
			}
		}
	}
	return created, nil
}

func (s *Store) scheduleShard(ctx context.Context, epoch int64, m model.Manifest, sh model.ManifestShard, now time.Time) (model.Attempt, bool, string, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return model.Attempt{}, false, "", err
	}
	defer func() { _ = tx.Rollback() }()

	var shardStatus string
	err = tx.QueryRowContext(ctx, `SELECT status FROM shards WHERE job_id=$1 AND id=$2 FOR UPDATE`, m.JobID, sh.ID).Scan(&shardStatus)
	if err != nil {
		return model.Attempt{}, false, "", err
	}
	if shardStatus == model.ShardSucceeded || shardStatus == model.ShardFailed {
		return model.Attempt{}, false, "", nil
	}
	var total, active, terminalFailures, usedSpeculative int
	var oldestStarted sql.NullTime
	var minProgress sql.NullFloat64
	err = tx.QueryRowContext(ctx, `
SELECT count(*),
       count(*) FILTER (WHERE status IN ('ready','running')),
       count(*) FILTER (WHERE status IN ('failed','rejected')),
       count(*) FILTER (WHERE kind='speculative'),
       min(started_at) FILTER (WHERE status='running'),
       min(progress) FILTER (WHERE status='running')
FROM attempts WHERE job_id=$1 AND shard_id=$2`, m.JobID, sh.ID).
		Scan(&total, &active, &terminalFailures, &usedSpeculative, &oldestStarted, &minProgress)
	if err != nil {
		return model.Attempt{}, false, "", err
	}

	kind, reason := "", ""
	switch {
	case total == 0:
		kind, reason = model.AttemptInitial, "initial dispatch"
	case active == 0 && terminalFailures > 0 && total < m.MaxAttemptsPerShard:
		kind, reason = model.AttemptRetry, "all visible attempts failed; bounded retry"
	case active > 0 && total < m.MaxAttemptsPerShard && usedSpeculative < m.MaxSpeculativeCopies:
		elapsed := time.Duration(0)
		if oldestStarted.Valid {
			elapsed = now.Sub(oldestStarted.Time)
		}
		threshold := time.Duration(m.SpecAfterMillis) * time.Millisecond
		progress := 1.0
		if minProgress.Valid {
			progress = minProgress.Float64
		}
		if elapsed >= threshold && progress <= 0.25 {
			kind = model.AttemptSpeculative
			reason = fmt.Sprintf("elapsed=%s >= spec_after=%s and min_progress=%.3f <= 0.250; budget used=%d/%d",
				elapsed.Truncate(time.Millisecond), threshold, progress, usedSpeculative, m.MaxSpeculativeCopies)
		}
	}

	if kind == "" {
		if active == 0 && terminalFailures > 0 && total >= m.MaxAttemptsPerShard {
			if err := markTerminalTx(ctx, tx, m.JobID, sh.ID, model.ShardFailed, model.FailureAttemptLimitExhausted,
				fmt.Sprintf("attempts=%d limit=%d", total, m.MaxAttemptsPerShard)); err != nil {
				return model.Attempt{}, false, "", err
			}
			if err := refreshJobStateTx(ctx, tx, m.JobID); err != nil {
				return model.Attempt{}, false, "", err
			}
			if err := tx.Commit(); err != nil {
				return model.Attempt{}, false, "", err
			}
			return model.Attempt{}, false, "", nil
		}
		return model.Attempt{}, false, "", nil
	}

	attemptNo := total + 1
	token, err := randomToken()
	if err != nil {
		return model.Attempt{}, false, "", err
	}
	if _, err := tx.ExecContext(ctx, `
INSERT INTO attempts(job_id,shard_id,attempt_no,run_epoch,token,kind,status,started_at,last_heartbeat_at)
VALUES ($1,$2,$3,$4,$5,$6,'ready',$7,$7)`, m.JobID, sh.ID, attemptNo, epoch, token, kind, now); err != nil {
		return model.Attempt{}, false, "", err
	}
	if _, err := tx.ExecContext(ctx, `UPDATE jobs SET status='running', updated_at=now() WHERE id=$1 AND status='queued'`, m.JobID); err != nil {
		return model.Attempt{}, false, "", err
	}
	if _, err := tx.ExecContext(ctx, `UPDATE shards SET status='running', updated_at=now() WHERE job_id=$1 AND id=$2`, m.JobID, sh.ID); err != nil {
		return model.Attempt{}, false, "", err
	}
	if err := tx.Commit(); err != nil {
		return model.Attempt{}, false, "", err
	}
	a := model.Attempt{JobID: m.JobID, ShardID: sh.ID, AttemptNo: attemptNo, Token: token, Kind: kind, Status: "ready", StartedAt: now, LastHeartbeatAt: now}
	return a, true, reason, nil
}

func markTerminalTx(ctx context.Context, tx *sql.Tx, jobID, shardID, status, category, reason string) error {
	if _, err := tx.ExecContext(ctx, `UPDATE shards SET status=$3, updated_at=now() WHERE job_id=$1 AND id=$2`, jobID, shardID, status); err != nil {
		return err
	}
	return insertEventTx(ctx, tx, model.Event{
		JobID: jobID, ShardID: shardID, Type: "shard_terminal", Status: status, Category: category, Reason: reason,
	})
}

func refreshJobStateTx(ctx context.Context, tx *sql.Tx, jobID string) error {
	var total, succeeded, failed int
	if err := tx.QueryRowContext(ctx, `
SELECT count(*),
       count(*) FILTER (WHERE status='succeeded'),
       count(*) FILTER (WHERE status='failed')
FROM shards WHERE job_id=$1`, jobID).Scan(&total, &succeeded, &failed); err != nil {
		return err
	}
	status := model.JobRunning
	if succeeded == total {
		status = model.JobSucceeded
	} else if failed > 0 {
		status = model.JobFailed
	}
	if _, err := tx.ExecContext(ctx, `UPDATE jobs SET status=$2, updated_at=now() WHERE id=$1`, jobID, status); err != nil {
		return err
	}
	return insertEventTx(ctx, tx, model.Event{
		JobID: jobID, Type: "job_progress", Status: status,
		Reason:  fmt.Sprintf("shards succeeded=%d failed=%d total=%d", succeeded, failed, total),
		Payload: mustJSON(map[string]int{"total": total, "succeeded": succeeded, "failed": failed}),
	})
}
