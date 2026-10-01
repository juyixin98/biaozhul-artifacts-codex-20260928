package replay_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
	"opp299/internal/replay"
)

type noopStore struct{}

func (noopStore) SaveCheckpoint(context.Context, protocol.Checkpoint) error { return nil }

func TestUnalignedSnapshotCarriesOnlyUnincorporatedRecordsAndRecoveryReplaysThem(t *testing.T) {
	cfg := protocol.Config{
		TaskID:                   "unaligned-unit",
		Mode:                     protocol.Unaligned,
		BackpressureLimit:        8,
		MaxConcurrentCheckpoints: 1,
	}
	store := newCheckpointCapture()
	runner, err := kernel.NewRunner(cfg, store)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- runner.Run(ctx) }()

	// The independent reference oracle is literal, not derived from runner state.
	mustPushRecord(t, runner, protocol.Record{Stream: protocol.StreamB, Key: "k1", Seq: 1, Value: 4})
	mustPushBarrier(t, runner, protocol.Barrier{ID: 1, Stream: protocol.StreamB, After: 1})

	// B is paused at its barrier. Two A records arrive before B is completed;
	// both remain unincorporated in the unaligned base state.
	mustPushRecord(t, runner, protocol.Record{Stream: protocol.StreamA, Key: "k1", Seq: 1, Value: 2})
	mustPushRecord(t, runner, protocol.Record{Stream: protocol.StreamA, Key: "k1", Seq: 2, Value: 3})

	// A record after B's barrier belongs to the next epoch and must be live.
	mustPushRecord(t, runner, protocol.Record{Stream: protocol.StreamB, Key: "k1", Seq: 2, Value: 40})
	mustPushBarrier(t, runner, protocol.Barrier{ID: 1, Stream: protocol.StreamA, After: 2})

	select {
	case cp := <-store.c:
		if got := cp.State.Counts["k1"]; got != (protocol.PairCount{A: 0, B: 4}) {
			t.Fatalf("snapshot base reference = %+v, want {A:0 B:4}", got)
		}
		inflightA := cp.InFlight[protocol.StreamA]
		if len(inflightA) != 2 || inflightA[0].Record.Seq != 1 || inflightA[1].Record.Seq != 2 {
			t.Fatalf("inflight A = %+v, want seq 1 and 2", inflightA)
		}
		if len(cp.InFlight[protocol.StreamB]) != 0 {
			t.Fatalf("inflight B = %+v, want empty", cp.InFlight[protocol.StreamB])
		}
		if cp.Metrics.BufferOccupancy != 2 {
			t.Fatalf("buffer occupancy at snapshot = %d, want 2 (A1,A2)", cp.Metrics.BufferOccupancy)
		}

		recovered, report, err := replay.Inflight(context.Background(), cfg, noopStore{}, cp)
		if err != nil {
			t.Fatalf("replay: %v", err)
		}
		if len(report.Inflight) != 2 {
			t.Fatalf("recovery report listed %d records, want 2", len(report.Inflight))
		}
		if got := recovered.StateSnapshot().Counts["k1"]; got != (protocol.PairCount{A: 5, B: 4}) {
			t.Fatalf("post-replay reference = %+v, want literal oracle {A:5 B:4}", got)
		}
		if recovered.SnapshotMetrics().Replayed != 2 {
			t.Fatalf("replayed count = %d, want 2", recovered.SnapshotMetrics().Replayed)
		}
	case <-time.After(time.Second):
		cancel()
		t.Fatalf("checkpoint was not stored; runner=%v metrics=%+v outputs=%+v", <-done, runner.SnapshotMetrics(), runner.Outputs())
	}
	cancel()
	select {
	case err := <-done:
		if err != nil && err.Error() != context.Canceled.Error() {
			t.Fatalf("runner error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("runner did not stop")
	}
}

func TestSnapshotInterruptionLeavesNoCheckpointAndAllowsRecoveryFromPreviousCompleteOne(t *testing.T) {
	cfg := protocol.Config{TaskID: "interrupt", Mode: protocol.Unaligned, BackpressureLimit: 8, MaxConcurrentCheckpoints: 1}
	complete := newCheckpointCapture()
	first, _ := kernel.NewRunner(cfg, complete)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- first.Run(ctx) }()
	mustPushRecord(t, first, protocol.Record{Stream: protocol.StreamA, Key: "k1", Seq: 1, Value: 1})
	mustPushBarrier(t, first, protocol.Barrier{ID: 1, Stream: protocol.StreamA, After: 1})
	mustPushRecord(t, first, protocol.Record{Stream: protocol.StreamB, Key: "k1", Seq: 1, Value: 9})
	mustPushBarrier(t, first, protocol.Barrier{ID: 1, Stream: protocol.StreamB, After: 1})
	var cp1 protocol.Checkpoint
	select {
	case cp1 = <-complete.c:
	case <-time.After(time.Second):
		t.Fatal("CP1 did not complete")
	}
	cancel()
	<-done

	interrupted := newCheckpointCapture()
	interrupted.failID = 2
	failedCfg := cfg
	failedCfg.FailCheckpointID = 2
	second, replayReport, err := replay.Inflight(context.Background(), failedCfg, interrupted, cp1)
	if err != nil {
		t.Fatal(err)
	}
	if len(replayReport.Inflight) != 1 || replayReport.Inflight[0].Stream != protocol.StreamB {
		t.Fatalf("CP1 unaligned inflight = %+v, want only B1", replayReport.Inflight)
	}
	failCtx, failCancel := context.WithCancel(context.Background())
	failDone := make(chan error, 1)
	go func() { failDone <- second.Run(failCtx) }()
	mustPushRecord(t, second, protocol.Record{Stream: protocol.StreamA, Key: "k1", Seq: 2, Value: 100})
	mustPushBarrier(t, second, protocol.Barrier{ID: 2, Stream: protocol.StreamA, After: 2})
	mustPushBarrier(t, second, protocol.Barrier{ID: 2, Stream: protocol.StreamB, After: 1})
	select {
	case err := <-failDone:
		if !errors.Is(err, kernel.ErrSnapshotFailureInjected) {
			t.Fatalf("CP2 error = %v, want injected snapshot failure", err)
		}
		if classified := kernel.Classify(err); classified.Category != kernel.CategoryStorage {
			t.Fatalf("failure category = %s, want storage_failure", classified.Category)
		}
	case <-time.After(time.Second):
		failCancel()
		t.Fatal("CP2 snapshot interruption did not stop the runner")
	}
	failCancel()
	if len(interrupted.c) != 0 {
		t.Fatal("failed CP2 was reported complete")
	}

	recovered, report, err := replay.Inflight(context.Background(), cfg, noopStore{}, cp1)
	if err != nil {
		t.Fatal(err)
	}
	if report.CheckpointID != 1 {
		t.Fatalf("recovery checkpoint = %d, want newest complete CP1", report.CheckpointID)
	}
	if got := recovered.StateSnapshot().Counts["k1"]; got != (protocol.PairCount{A: 1, B: 9}) {
		t.Fatalf("reference after CP1 recovery = %+v, want {A:1 B:9}", got)
	}
}

type checkpointCapture struct {
	c      chan protocol.Checkpoint
	failID int64
}

func newCheckpointCapture() *checkpointCapture {
	return &checkpointCapture{c: make(chan protocol.Checkpoint, 4)}
}

func (s *checkpointCapture) SaveCheckpoint(_ context.Context, cp protocol.Checkpoint) error {
	if s.failID != 0 && cp.ID == s.failID {
		return &kernel.ClassifiedError{Category: kernel.CategoryStorage, Err: kernel.ErrSnapshotFailureInjected}
	}
	s.c <- cp
	return nil
}

func mustPushRecord(t *testing.T, r *kernel.Runner, rec protocol.Record) {
	t.Helper()
	ctx, done := context.WithTimeout(context.Background(), time.Second)
	defer done()
	if err := r.PushRecord(ctx, rec); err != nil {
		t.Fatal(err)
	}
}

func mustPushBarrier(t *testing.T, r *kernel.Runner, b protocol.Barrier) {
	t.Helper()
	ctx, done := context.WithTimeout(context.Background(), time.Second)
	defer done()
	if err := r.PushBarrier(ctx, b); err != nil {
		t.Fatal(err)
	}
}
