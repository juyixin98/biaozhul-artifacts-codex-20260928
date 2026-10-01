package kernel_test

import (
	"context"
	"errors"
	"testing"

	"checkpointlab/internal/kernel"
	"checkpointlab/internal/protocol"
	"checkpointlab/internal/source"
	"checkpointlab/internal/storage"
)

func TestAlignedBlocksFastStreamAndCommitsReferenceCount(t *testing.T) {
	sc := mustScenario(t, "../../testdata/scenarios/aligned-speed-diff.json")
	store := storage.NewMemoryStore()
	ctx := context.Background()
	if err := store.CreateJob(ctx, sc.Name, sc.Mode, nil); err != nil {
		t.Fatal(err)
	}
	op, err := kernel.NewOperator(kernel.Config{
		JobID: sc.Name, Mode: sc.Mode, GateCapacity: sc.GateCapacity, InflightLimit: sc.InflightLimit,
	}, store)
	if err != nil {
		t.Fatal(err)
	}

	var blocked *protocol.TypedError
	for _, step := range sc.Steps {
		_, err := op.Process(ctx, step.Event(), "req-"+step.Name)
		if step.Name == "a2-post-barrier" {
			gateErr := op.Gate("A").TrySend(step.Event())
			if !errors.As(gateErr, &blocked) || blocked.Category != protocol.CategoryAlignmentBlocked {
				t.Fatalf("expected ALIGNED_INPUT_BLOCKED, got %v", gateErr)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", step.Name, err)
		}
	}

	metrics := op.Metrics()
	if metrics.CommittedEpoch != 1 || metrics.ActiveEpoch != 0 {
		t.Fatalf("committed=%d active=%d", metrics.CommittedEpoch, metrics.ActiveEpoch)
	}
	if metrics.PeakInflight != 0 || metrics.Gates["A"].BlockedAttempts != 1 {
		t.Fatalf("metrics=%+v", metrics)
	}
	outs, err := store.ListOutputs(ctx, sc.Name, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 3 {
		t.Fatalf("committed outputs=%d want 3", len(outs))
	}
	state := sumCommitted(outs)
	if state["x"] != 5 || state["y"] != 2 {
		t.Fatalf("state=%v want x=5,y=2", state)
	}
}

func TestUnalignedBuffersOnlyUnclosedRecordsAndRecoversWithoutRecomputingInflights(t *testing.T) {
	sc := mustScenario(t, "../../testdata/scenarios/unaligned-speed-diff.json")
	store := storage.NewMemoryStore()
	ctx := context.Background()
	if err := store.CreateJob(ctx, sc.Name, sc.Mode, nil); err != nil {
		t.Fatal(err)
	}
	op, err := kernel.NewOperator(kernel.Config{
		JobID: sc.Name, Mode: sc.Mode, GateCapacity: sc.GateCapacity, InflightLimit: sc.InflightLimit,
	}, store)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range sc.Steps {
		if _, err := op.Process(ctx, step.Event(), "req-"+step.Name); err != nil {
			t.Fatalf("%s: %v", step.Name, err)
		}
	}
	metrics := op.Metrics()
	if metrics.CommittedEpoch != 1 || metrics.PeakInflight != 1 {
		t.Fatalf("metrics=%+v", metrics)
	}

	snap, ok, err := store.GetCheckpoint(ctx, sc.Name, 1)
	if err != nil || !ok {
		t.Fatalf("checkpoint err=%v ok=%v", err, ok)
	}
	if len(snap.Inflights["B"]) != 1 {
		t.Fatalf("inflights=%+v", snap.Inflights)
	}
	if snap.Base["x"] != 1 || snap.Base["y"] != 2 {
		t.Fatalf("base=%v want x=1 y=2", snap.Base)
	}
	all, _ := store.ListOutputs(ctx, sc.Name, false)
	committed, _ := store.ListOutputs(ctx, sc.Name, true)
	if len(all) != 4 || len(committed) != 3 {
		t.Fatalf("all=%d committed=%d", len(all), len(committed))
	}

	restored, err := kernel.NewOperator(kernel.Config{
		JobID: sc.Name, Mode: sc.Mode, GateCapacity: sc.GateCapacity, InflightLimit: sc.InflightLimit,
	}, store)
	if err != nil {
		t.Fatal(err)
	}
	point, err := restored.Restore(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if point.ReplayStart["A"] != 2 || point.ReplayStart["B"] != 1 {
		t.Fatalf("replay start=%+v", point.ReplayStart)
	}
	state := restored.SnapshotState()
	if state["x"] != 5 || state["y"] != 2 {
		t.Fatalf("restored=%v want x=5 y=2", state)
	}
	if restored.Metrics().PeakInflight != 0 {
		t.Fatalf("restored operator must not count snapshot inflights as live buffered records: %+v", restored.Metrics())
	}
	_, err = restored.Process(ctx, protocol.RecordEvent(protocol.Record{
		Stream: "B", Offset: 0, Epoch: 1, Key: "x", Value: 4,
	}), "req-duplicate")
	var typed *protocol.TypedError
	if !errors.As(err, &typed) || typed.Category != protocol.CategoryReplayDuplicate {
		t.Fatalf("expected replay duplicate category, got %v", err)
	}
}

func TestOutOfOrderBarrierRejectedBySpecificCategory(t *testing.T) {
	store := storage.NewMemoryStore()
	ctx := context.Background()
	if err := store.CreateJob(ctx, "barrier-order", protocol.ModeAligned, nil); err != nil {
		t.Fatal(err)
	}
	op, _ := kernel.NewOperator(kernel.Config{
		JobID: "barrier-order", Mode: protocol.ModeAligned, GateCapacity: 2, InflightLimit: 2,
	}, store)
	if _, err := op.Process(ctx, protocol.BarrierEvent(protocol.Barrier{Stream: "A", Epoch: 2, Offset: 0}), "r1"); err != nil {
		t.Fatal(err)
	}
	_, err := op.Process(ctx, protocol.BarrierEvent(protocol.Barrier{Stream: "B", Epoch: 1, Offset: 0}), "r2")
	var typed *protocol.TypedError
	if !errors.As(err, &typed) || typed.Category != protocol.CategoryCheckpointScope {
		t.Fatalf("expected CONCURRENT_CHECKPOINT_REJECTED for older different barrier, got %v", err)
	}

	newStore := storage.NewMemoryStore()
	_ = newStore.CreateJob(ctx, "backward", protocol.ModeAligned, nil)
	op2, _ := kernel.NewOperator(kernel.Config{
		JobID: "backward", Mode: protocol.ModeAligned, GateCapacity: 2, InflightLimit: 2,
	}, newStore)
	_, _ = op2.Process(ctx, protocol.BarrierEvent(protocol.Barrier{Stream: "A", Epoch: 1, Offset: 0}), "a1")
	_, _ = op2.Process(ctx, protocol.BarrierEvent(protocol.Barrier{Stream: "B", Epoch: 1, Offset: 0}), "b1")
	_, err = op2.Process(ctx, protocol.BarrierEvent(protocol.Barrier{Stream: "A", Epoch: 1, Offset: 0}), "a-old")
	if !errors.As(err, &typed) || typed.Category != protocol.CategoryBarrierOrder {
		t.Fatalf("expected BARRIER_ORDER_VIOLATION, got %v", err)
	}
}

func mustScenario(t *testing.T, path string) source.Scenario {
	t.Helper()
	sc, err := source.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return sc
}

func sumCommitted(outs []storage.Output) map[string]int64 {
	result := map[string]int64{}
	for _, out := range outs {
		if out.Committed {
			result[out.Key] += out.Value
		}
	}
	return result
}
