package storage

import (
	"context"

	"opp299/internal/protocol"
)

type CheckpointStore interface {
	Init(ctx context.Context) error
	SaveCheckpoint(ctx context.Context, cp protocol.Checkpoint) error
	GetLatest(ctx context.Context, taskID string) (protocol.Checkpoint, error)
	GetCheckpoint(ctx context.Context, taskID string, id int64) (protocol.Checkpoint, error)
	ListCheckpoints(ctx context.Context, taskID string, limit int) ([]protocol.Checkpoint, error)
}
