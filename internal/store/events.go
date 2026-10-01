package store

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5/pgconn"
)

type eventExecer interface {
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
}

func insertEvent(ctx context.Context, tx eventExecer, jobID, shardID, attemptID, eventType string, version int, epoch string, payload map[string]any) error {
	if payload == nil {
		payload = map[string]any{}
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `
INSERT INTO events (job_id, shard_id, attempt_id, event_type, version, epoch, payload)
VALUES ($1,$2,$3,$4,$5,$6,$7)`,
		jobID, shardID, attemptID, eventType, version, epoch, raw)
	return err
}
