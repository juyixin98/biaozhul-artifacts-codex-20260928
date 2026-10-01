package engine_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
	"opp299/internal/replay"
	"opp299/internal/storage"
)

type channelStore struct {
	c     chan protocol.Checkpoint
	saved map[int64]protocol.Checkpoint
}

func newChannelStore() *channelStore {
	return &channelStore{c: make(chan protocol.Checkpoint, 4), saved: map[int64]protocol.Checkpoint{}}
}

func (s *channelStore) SaveCheckpoint(_ context.Context, cp protocol.Checkpoint) error {
	s.saved[cp.ID] = cp
	s.c <- cp
	return nil
}

func (s *channelStore) GetCheckpoint(_ context.Context, _ string, id int64) (protocol.Checkpoint, error) {
	if cp, ok := s.saved[id]; ok {
		return cp, nil
	}
	select {
	case cp := <-s.c:
		s.saved[cp.ID] = cp
		if cp.ID == id {
			return cp, nil
		}
	case <-time.After(time.Second):
		return protocol.Checkpoint{}, errors.New("checkpoint not found")
	}
	return s.GetCheckpoint(context.Background(), "", id)
}

type failCP2Store struct {
	inner *channelStore
}

func (s failCP2Store) SaveCheckpoint(ctx context.Context, cp protocol.Checkpoint) error {
	if cp.ID == 2 {
		return &kernel.ClassifiedError{Category: kernel.CategoryStorage, Err: kernel.ErrSnapshotFailureInjected}
	}
	return s.inner.SaveCheckpoint(ctx, cp)
}

func (s failCP2Store) GetCheckpoint(ctx context.Context, taskID string, id int64) (protocol.Checkpoint, error) {
	return s.inner.GetCheckpoint(ctx, taskID, id)
}

func runLoop(t *testing.T, runner *kernel.Runner) (context.CancelFunc, chan error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- runner.Run(ctx) }()
	return cancel, done
}

func pushRecord(t *testing.T, runner *kernel.Runner, r protocol.Record) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := runner.PushRecord(ctx, r); err != nil {
		t.Fatalf("push %s/%d: %v", r.Stream, r.Seq, err)
	}
}

func pushBarrier(t *testing.T, runner *kernel.Runner, b protocol.Barrier) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := runner.PushBarrier(ctx, b); err != nil {
		t.Fatalf("push barrier %d/%s: %v", b.ID, b.Stream, err)
	}
}

func rec(stream, key string, seq int64, value int64) protocol.Record {
	return protocol.Record{Stream: stream, Key: key, Seq: seq, Value: value}
}

func waitCheckpoint(t *testing.T, ch <-chan protocol.Checkpoint) protocol.Checkpoint {
	t.Helper()
	select {
	case cp := <-ch:
		return cp
	case <-time.After(time.Second):
		t.Fatal("checkpoint was not stored")
		return protocol.Checkpoint{}
	}
}

func cfg(mode protocol.Mode, failID int64) protocol.Config {
	return protocol.Config{
		TaskID:                   "speed-diff",
		Mode:                     mode,
		BackpressureLimit:        8,
		MaxConcurrentCheckpoints: 1,
		FailCheckpointID:         failID,
	}
}

func stageCP1(t *testing.T, mode protocol.Mode, store kernel.Store) (context.CancelFunc, chan error, *kernel.Runner, protocol.Checkpoint) {
	runner, err := kernel.NewRunner(cfg(mode, 2), store)
	if err != nil {
		t.Fatal(err)
	}
	// Slow B reaches its CP1 boundary first.
	pushRecord(t, runner, rec(protocol.StreamB, "k1", 1, 100))
	if _, err := runner.ProcessPending(context.Background()); err != nil {
		t.Fatal(err)
	}
	pushBarrier(t, runner, protocol.Barrier{ID: 1, Stream: protocol.StreamB, After: 1})
	if _, err := runner.ProcessPending(context.Background()); err != nil {
		t.Fatal(err)
	}

	// Fast A has three CP1 records queued before the matching barrier.
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 1, 1))
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 2, 1))
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 3, 1))

	// B already has one post-CP1 record waiting when A closes CP1.
	pushRecord(t, runner, rec(protocol.StreamB, "k1", 2, 100))
	pushBarrier(t, runner, protocol.Barrier{ID: 1, Stream: protocol.StreamA, After: 3})
	if _, err := runner.ProcessPending(context.Background()); err != nil {
		t.Fatal(err)
	}

	checkpointStore := store.(interface {
		GetCheckpoint(context.Context, string, int64) (protocol.Checkpoint, error)
	})
	cp, err := checkpointStore.GetCheckpoint(context.Background(), "speed-diff", 1)
	if err != nil {
		t.Fatal(err)
	}
	cancel, done := runLoop(t, runner)
	return cancel, done, runner, cp
}

func stageFailedCP2(t *testing.T, cancel context.CancelFunc, done chan error, runner *kernel.Runner) {
	t.Helper()
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 4, 1))
	pushBarrier(t, runner, protocol.Barrier{ID: 2, Stream: protocol.StreamA, After: 4})
	pushBarrier(t, runner, protocol.Barrier{ID: 2, Stream: protocol.StreamB, After: 2})
	select {
	case err := <-done:
		if !errors.Is(err, kernel.ErrSnapshotFailureInjected) {
			t.Fatalf("CP2 error = %v, want injected storage failure", err)
		}
		if kernel.Classify(err).Category != kernel.CategoryStorage {
			t.Fatalf("CP2 category = %s, want storage_failure", kernel.Classify(err).Category)
		}
	case <-time.After(time.Second):
		cancel()
		t.Fatal("CP2 failure did not stop the runner")
	}
	cancel()
}

func recoverToCP2(t *testing.T, mode protocol.Mode, cp protocol.Checkpoint) (*kernel.Runner, replay.Report, protocol.Checkpoint) {
	t.Helper()
	store := newChannelStore()
	runner, report, err := replay.Inflight(context.Background(), cfg(mode, 0), store, cp)
	if err != nil {
		t.Fatal(err)
	}
	cancel, done := runLoop(t, runner)
	defer cancel()

	// A CP2 event was lost with the interrupted process; B2 may also have been
	// accepted into the queue. Send only what the checkpoint did not include.
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 4, 1))
	pushBarrier(t, runner, protocol.Barrier{ID: 2, Stream: protocol.StreamA, After: 4})

	// CP1 did not include records after its barrier in either mode. Recovery
	// replays attached pre-barrier records, then resumes both source logs.
	pushRecord(t, runner, rec(protocol.StreamB, "k1", 2, 100))
	pushBarrier(t, runner, protocol.Barrier{ID: 2, Stream: protocol.StreamB, After: 2})
	cp2 := waitCheckpoint(t, store.c)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("recovered run stopped early: %v", err)
		}
	case <-time.After(20 * time.Millisecond):
	}
	return runner, report, cp2
}

func TestAlignedAndUnalignedRecoveryProduceSameReferenceCountsWithDifferentBuffers(t *testing.T) {
	alignedStore := failCP2Store{inner: newChannelStore()}
	alignedCancel, alignedDone, alignedRunner, alignedCP1 := stageCP1(t, protocol.Aligned, alignedStore)
	stageFailedCP2(t, alignedCancel, alignedDone, alignedRunner)

	unalignedStore := failCP2Store{inner: newChannelStore()}
	unalignedCancel, unalignedDone, unalignedRunner, unalignedCP1 := stageCP1(t, protocol.Unaligned, unalignedStore)
	stageFailedCP2(t, unalignedCancel, unalignedDone, unalignedRunner)

	alignedInflight := len(alignedCP1.InFlight[protocol.StreamA]) + len(alignedCP1.InFlight[protocol.StreamB])
	unalignedInflight := len(unalignedCP1.InFlight[protocol.StreamA]) + len(unalignedCP1.InFlight[protocol.StreamB])
	if alignedInflight != 0 {
		t.Fatalf("aligned CP1 inflight = %d, want 0", alignedInflight)
	}
	if unalignedInflight != 3 {
		t.Fatalf("unaligned CP1 inflight = %d, want A1,A2,A3 = 3 records", unalignedInflight)
	}
	if got := alignedCP1.State.Counts["k1"]; got != (protocol.PairCount{A: 3, B: 100}) {
		t.Fatalf("aligned CP1 state = %+v, want literal oracle {A:3 B:100}", got)
	}
	if got := unalignedCP1.State.Counts["k1"]; got != (protocol.PairCount{A: 0, B: 100}) {
		t.Fatalf("unaligned CP1 state = %+v, want literal oracle {A:0 B:100}", got)
	}

	alignedRunner, alignedReport, alignedCP2 := recoverToCP2(t, protocol.Aligned, alignedCP1)
	unalignedRunner, unalignedReport, unalignedCP2 := recoverToCP2(t, protocol.Unaligned, unalignedCP1)

	want := protocol.PairCount{A: 4, B: 200}
	if got := alignedRunner.StateSnapshot().Counts["k1"]; got != want {
		t.Fatalf("aligned recovered reference = %+v, want %+v", got, want)
	}
	if got := unalignedRunner.StateSnapshot().Counts["k1"]; got != want {
		t.Fatalf("unaligned recovered reference = %+v, want %+v", got, want)
	}
	if len(alignedReport.Inflight) != 0 || alignedReport.Mode != protocol.Aligned {
		t.Fatalf("aligned replay report = %+v, want no inflight in aligned mode", alignedReport)
	}
	if len(unalignedReport.Inflight) != unalignedInflight || unalignedReport.Mode != protocol.Unaligned {
		t.Fatalf("unaligned replay report = %+v, want %d inflight", unalignedReport, unalignedInflight)
	}
	if unalignedRunner.SnapshotMetrics().Replayed != int64(unalignedInflight) {
		t.Fatalf("replayed metric = %d, want %d", unalignedRunner.SnapshotMetrics().Replayed, unalignedInflight)
	}
	if alignedCP2.Metrics.BufferOccupancy >= unalignedCP2.Metrics.BufferOccupancy {
		t.Fatalf("buffer occupancy aligned=%d unaligned=%d; unaligned must show carried records",
			alignedCP2.Metrics.BufferOccupancy, unalignedCP2.Metrics.BufferOccupancy)
	}

	// Guard against the forbidden implementation: unaligned recovery must not
	// apply named inflight records twice.
	if got := unalignedRunner.StateSnapshot().Counts["k1"]; got.A > want.A || got.B > want.B {
		t.Fatalf("inflight records were double-counted: got %+v", got)
	}
}

var _ = storage.NewMemoryStore
