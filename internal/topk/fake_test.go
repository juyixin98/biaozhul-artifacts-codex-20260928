package topk

import (
	"context"
	"fmt"
	"time"

	"topk-demo/internal/protocol"
)

type fakeShard struct {
	items           []protocol.Candidate
	boundDelay      time.Duration
	scanDelay       time.Duration
	boundOverride   *protocol.Candidate
	boundGeneration string
	scanGeneration  string
	scanUpperBogus  *protocol.Candidate
}

type fakeClient map[string]*fakeShard

func newFakeClient(data map[string][]protocol.Candidate) fakeClient {
	out := fakeClient{}
	for id, items := range data {
		out[id] = &fakeShard{items: append([]protocol.Candidate(nil), items...)}
	}
	return out
}

func (c fakeClient) Bound(ctx context.Context, req protocol.BoundRequest) (*protocol.BoundResponse, error) {
	shard := c[req.ShardID]
	if shard == nil {
		return nil, &ShardCallError{Category: protocol.FailureInvalidShard, Message: "missing shard"}
	}
	if shard.boundDelay > 0 && !sleep(ctx, shard.boundDelay) {
		return nil, &ShardCallError{Category: protocol.FailureTimeout, Message: "bound deadline exceeded"}
	}
	idx := indexAt(shard.items, req.Cursor, false)
	resp := &protocol.BoundResponse{
		Generation:  req.Generation,
		ShardID:     req.ShardID,
		Cursor:      req.Cursor,
		BoundProbes: 1,
	}
	if shard.boundGeneration != "" {
		resp.Generation = shard.boundGeneration
	}
	if idx >= len(shard.items) {
		resp.Exhausted = true
		return resp, nil
	}
	upper := shard.items[idx]
	if shard.boundOverride != nil {
		upper = *shard.boundOverride
	}
	resp.Upper = &upper
	resp.ReadRows = 1
	return resp, nil
}

func (c fakeClient) Scan(ctx context.Context, req protocol.ScanRequest) (*protocol.ScanResponse, error) {
	shard := c[req.ShardID]
	if shard == nil {
		return nil, &ShardCallError{Category: protocol.FailureInvalidShard, Message: "missing shard"}
	}
	if shard.scanDelay > 0 && !sleep(ctx, shard.scanDelay) {
		return nil, &ShardCallError{Category: protocol.FailureTimeout, Message: "scan deadline exceeded"}
	}
	initial := req.Cursor == ""
	idx := indexAt(shard.items, req.Cursor, !initial)
	resp := &protocol.ScanResponse{
		Generation:         req.Generation,
		ShardID:            req.ShardID,
		Cursor:             req.Cursor,
		Items:              []protocol.Candidate{},
		ItemsIncludedUpper: !initial && idx < len(shard.items),
	}
	if !initial && idx < len(shard.items) {
		requested := shard.items[idx]
		resp.RequestedUpper = &requested
	}
	if shard.scanGeneration != "" {
		resp.Generation = shard.scanGeneration
	}
	if idx >= len(shard.items) {
		resp.Exhausted = true
		return resp, nil
	}
	end := idx + req.PageSize
	if end >= len(shard.items) {
		resp.Items = shard.items[idx:]
		resp.ReadRows = int64(len(resp.Items))
		resp.ReturnedRows = int64(len(resp.Items))
		resp.Exhausted = true
		return resp, nil
	}
	resp.Items = shard.items[idx:end]
	upper := shard.items[end]
	if shard.scanUpperBogus != nil {
		upper = *shard.scanUpperBogus
	}
	resp.NextUpper = &upper
	resp.NextCursorAt = &upper
	cursorTarget := shard.items[end]
	resp.NextCursor = encodeCursor(cursorTarget.Score, cursorTarget.ItemID)
	resp.ReadRows = int64(req.PageSize)
	if !initial {
		resp.ReadRows++
	}
	resp.ReturnedRows = int64(req.PageSize)
	resp.BoundProbes = 1
	return resp, nil
}

func indexAt(items []protocol.Candidate, cursor string, include bool) int {
	if cursor == "" {
		return 0
	}
	var score int64
	var id string
	if _, err := fmt.Sscanf(cursor, "%d/%s", &score, &id); err != nil {
		return len(items)
	}
	for i, item := range items {
		if item.Score == score && item.ItemID == id {
			if include {
				return i
			}
			return i + 1
		}
	}
	return len(items)
}

func encodeCursor(score int64, id string) string { return fmt.Sprintf("%d/%s", score, id) }

func sleep(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
