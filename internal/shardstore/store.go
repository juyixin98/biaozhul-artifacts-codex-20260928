package shardstore

import (
	"context"

	"topk-demo/internal/protocol"
)

type Store interface {
	Bound(ctx context.Context, generation, shardID, rawCursor string, cursor Cursor) (*protocol.BoundResponse, error)
	Scan(ctx context.Context, generation, shardID, rawCursor string, cursor Cursor, pageSize int) (*protocol.ScanResponse, error)
	Close()
}

func cursorFor(score int64, itemID string) Cursor {
	return Cursor{ResumeScore: score, ResumeItemID: itemID, Include: true, Version: 1}
}
