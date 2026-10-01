//go:build integration

package storage_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
	"opp299/internal/storage"
)

func TestPostgresAtomicCheckpointAndOutputs(t *testing.T) {
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		dsn = "host=/var/run/postgresql dbname=opp299 user=admin sslmode=disable"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	store, err := storage.NewPostgresStore(dsn, "../../migrations/001_checkpoints.sql")
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := store.Ping(ctx); err != nil {
		t.Skipf("postgres is not available: %v", err)
	}
	if err := store.Init(ctx); err != nil {
		t.Fatal(err)
	}

	taskID := fmt.Sprintf("integration-%d", time.Now().UnixNano())
	cp := protocol.Checkpoint{
		TaskID:     taskID,
		ID:         1,
		Mode:       protocol.Unaligned,
		Status:     protocol.CheckpointComplete,
		State:      &protocol.State{Counts: map[string]protocol.PairCount{"k1": {A: 3, B: 100}}},
		InFlight:   map[string][]protocol.InFlight{protocol.StreamA: {{Record: protocol.Record{Stream: protocol.StreamA, Key: "k1", Seq: 1, Value: 1}}}},
		Positions:  map[string]protocol.Position{protocol.StreamA: {Stream: protocol.StreamA, BarrierAfter: 3}, protocol.StreamB: {Stream: protocol.StreamB, BarrierAfter: 1}},
		Metrics:    protocol.Metrics{Processed: map[string]int64{protocol.StreamA: 3, protocol.StreamB: 1}, BufferOccupancy: 2, MaxBuffer: 4},
		Outputs:    []protocol.Output{{ID: "A-k1-00000000000000000001", Stream: protocol.StreamA, Key: "k1", Seq: 1, Epoch: 1}},
		StartedAt:  "2026-09-29T00:00:00Z",
		FinishedAt: "2026-09-29T00:00:01Z",
	}
	if err := store.SaveCheckpoint(ctx, cp); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveCheckpoint(ctx, cp); err != nil {
		t.Fatalf("idempotent checkpoint retry failed: %v", err)
	}
	old := cp
	old.ID = 0
	if err := store.SaveCheckpoint(ctx, old); !errors.Is(err, kernel.ErrBarrierOutOfOrder) {
		t.Fatalf("old checkpoint error = %v, want monotonic rejection", err)
	}

	loaded, err := store.GetLatest(ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.State.Counts["k1"] != (protocol.PairCount{A: 3, B: 100}) {
		t.Fatalf("loaded state = %+v", loaded.State.Counts)
	}
	if len(loaded.InFlight[protocol.StreamA]) != 1 || len(loaded.Outputs) != 1 {
		t.Fatalf("loaded inflight=%d outputs=%d, want 1 each", len(loaded.InFlight[protocol.StreamA]), len(loaded.Outputs))
	}
	if loaded.Metrics.BufferOccupancy != 2 || loaded.Metrics.Processed[protocol.StreamB] != 1 {
		t.Fatalf("loaded metrics = %+v", loaded.Metrics)
	}
}
