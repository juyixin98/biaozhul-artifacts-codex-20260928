package state

import (
	"context"

	"topk-demo/internal/protocol"
)

type RunStore interface {
	SaveRun(ctx context.Context, run protocol.RunRecord) error
	SaveEvent(ctx context.Context, event protocol.EventRecord) error
	GetRun(ctx context.Context, runID string) (*protocol.RunRecord, []protocol.EventRecord, error)
	Close()
}
