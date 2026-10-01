package storage_test

import (
	"context"
	"errors"
	"testing"

	"checkpointlab/internal/protocol"
	"checkpointlab/internal/storage"
)

func TestMemoryStoreActiveScopeAndRestoresOnlySealedBoundary(t *testing.T) {
	store := storage.NewMemoryStore()
	ctx := context.Background()
	if err := store.CreateJob(ctx, "job", protocol.ModeUnaligned, nil); err != nil {
		t.Fatal(err)
	}
	snap := newSnap("job", 1, protocol.ModeUnaligned)
	if err := store.StartCheckpoint(ctx, snap); err != nil {
		t.Fatal(err)
	}
	err := store.StartCheckpoint(ctx, newSnap("job", 2, protocol.ModeUnaligned))
	var typed *protocol.TypedError
	if !errors.As(err, &typed) || typed.Category != protocol.CategoryCheckpointScope {
		t.Fatalf("expected scope rejection, got %v", err)
	}

	_ = store.StageOutput(ctx, storage.Output{
		JobID: "job", Epoch: 1, Stream: "A", Offset: 0, Key: "x", Value: 1,
		Committed: false, OutputID: "job:A:0",
	})
	point, err := store.Restore(ctx, "job")
	if err != nil {
		t.Fatal(err)
	}
	if point.CommittedEpoch != 0 {
		t.Fatalf("active checkpoint must not restore: %+v", point)
	}
	if outs, _ := store.ListOutputs(ctx, "job", false); len(outs) != 0 {
		t.Fatalf("speculative outputs should be removed: %d", len(outs))
	}

	snap2 := newSnap("job", 2, protocol.ModeUnaligned)
	// The interrupted epoch is preserved as SUPERSEDED. The next barrier ID is
	// strictly greater, matching the monotonic checkpoint contract.
	if err := store.StartCheckpoint(ctx, snap2); err != nil {
		t.Fatal(err)
	}
	if err := store.ObserveBarrier(ctx, "job", 2, snap2.AttemptID, protocol.Barrier{Stream: "B", Epoch: 2, Offset: 1}); err != nil {
		t.Fatal(err)
	}
	ref := storage.RecordRef{Stream: "B", Offset: 0, Epoch: 1, Key: "x", Value: 4}
	if err := store.AppendInflight(ctx, "job", 2, snap2.AttemptID, ref); err != nil {
		t.Fatal(err)
	}
	_ = store.StageOutput(ctx, storage.Output{
		JobID: "job", Epoch: 2, Stream: "B", Offset: 0, Key: "x", Value: 4,
		Committed: false, OutputID: "job:B:0",
	})
	_, err = store.SealCheckpoint(ctx, "job", 2, snap2.AttemptID,
		storage.State{"x": 1, "y": 2}, []storage.RecordRef{ref})
	if err != nil {
		t.Fatal(err)
	}
	_ = store.StageOutput(ctx, storage.Output{
		JobID: "job", Epoch: 3, Stream: "A", Offset: 1, Key: "x", Value: 9,
		Committed: false, OutputID: "job:A:1",
	})
	committed, _ := store.ListOutputs(ctx, "job", true)
	if len(committed) != 1 {
		t.Fatalf("seal must atomically commit epoch<=1 outputs: %+v", committed)
	}
	point2, err := store.Restore(ctx, "job")
	if err != nil || point2.CommittedEpoch != 2 {
		t.Fatalf("restore=%+v err=%v", point2, err)
	}
	remaining, _ := store.ListOutputs(ctx, "job", false)
	if len(remaining) != 1 {
		t.Fatalf("epoch-2 speculative output must be removed, got %+v", remaining)
	}
}

func newSnap(jobID string, epoch int64, mode protocol.Mode) storage.Snapshot {
	return storage.Snapshot{
		JobID:     jobID,
		Epoch:     epoch,
		AttemptID: "attempt",
		Mode:      mode,
		Base:      storage.State{},
		Inflights: map[string][]storage.RecordRef{},
		Targets: map[string]storage.BarrierTarget{
			"A": {Offset: 1, Seen: true},
		},
		SeenBarriers: map[string]bool{"A": true},
		BaseOffsets:  map[string]int64{"A": 1, "B": 0},
	}
}
