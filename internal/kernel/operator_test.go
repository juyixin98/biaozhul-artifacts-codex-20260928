package kernel_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
)

func testConfig(mode protocol.Mode) protocol.Config {
	return protocol.Config{
		TaskID:                   "unit",
		Mode:                     mode,
		BackpressureLimit:        4,
		MaxConcurrentCheckpoints: 1,
	}
}

func rec(stream, key string, seq int64, value int64) protocol.Record {
	return protocol.Record{Stream: stream, Key: key, Seq: seq, Value: value}
}

func barrier(id int64, stream string, after int64) protocol.Barrier {
	return protocol.Barrier{ID: id, Stream: stream, After: after}
}

func waitFor(t *testing.T, timeout time.Duration, fn func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if fn() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("condition was not reached within %s", timeout)
}

func runUntil(t *testing.T, runner *kernel.Runner, fn func() bool) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- runner.Run(ctx) }()
	waitFor(t, time.Second, fn)
	cancel()
	select {
	case err := <-done:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("runner stopped with unexpected error: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("runner did not stop after cancellation")
	}
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

func TestAlignedCheckpointDrainsQueuesAndBlocksBehindBarrier(t *testing.T) {
	store := newRecordingStore()
	runner, err := kernel.NewRunner(testConfig(protocol.Aligned), store)
	if err != nil {
		t.Fatal(err)
	}

	// A races ahead: B has only its first two records when A's barrier arrives.
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 1, 1))
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 2, 1))
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 3, 1))
	pushBarrier(t, runner, barrier(1, protocol.StreamA, 3))
	pushRecord(t, runner, rec(protocol.StreamB, "x", 1, 7))
	pushRecord(t, runner, rec(protocol.StreamB, "x", 2, 7))

	runUntil(t, runner, func() bool {
		m := runner.SnapshotMetrics()
		return m.Processed[protocol.StreamA] == 3 && m.Processed[protocol.StreamB] == 2
	})

	pushRecord(t, runner, rec(protocol.StreamB, "k1", 3, 5))
	pushBarrier(t, runner, barrier(1, protocol.StreamB, 3))
	runUntil(t, runner, func() bool { return store.len() == 1 })

	cp := store.get(0)
	if got := len(cp.InFlight[protocol.StreamA]) + len(cp.InFlight[protocol.StreamB]); got != 0 {
		t.Fatalf("aligned checkpoint inflight = %d, want 0", got)
	}
	if got := cp.State.Counts["k1"]; got != (protocol.PairCount{A: 3, B: 5}) {
		t.Fatalf("k1 counts = %+v, want {A:3 B:5}", got)
	}
	// This record is after barrier A=3. It cannot be incorporated in CP1 and,
	// in aligned mode, the queue is blocked behind the barrier boundary.
	pushRecord(t, runner, rec(protocol.StreamA, "k1", 4, 1))
	runUntil(t, runner, func() bool {
		return runner.SnapshotMetrics().BufferOccupancy == 1
	})
	state := runner.StateSnapshot()
	if got := state.Counts["k1"]; got.A != 3 {
		t.Fatalf("post-barrier A record was processed during aligned snapshot, A=%d", got.A)
	}
}

func TestRejectsOutOfOrderAndDuplicateBarriers(t *testing.T) {
	tests := []struct {
		name     string
		barriers []protocol.Barrier
		want     error
	}{
		{
			name: "backwards barrier id",
			barriers: []protocol.Barrier{
				{ID: 1, Stream: protocol.StreamA, After: 0},
				{ID: 0, Stream: protocol.StreamB, After: 0, OutOfOrder: true},
			},
			want: kernel.ErrBarrierOutOfOrder,
		},
		{
			name: "skipped barrier id",
			barriers: []protocol.Barrier{
				{ID: 2, Stream: protocol.StreamA, After: 0},
			},
			want: kernel.ErrBarrierGap,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner, _ := kernel.NewRunner(testConfig(protocol.Aligned), nil)
			ctx, cancel := context.WithCancel(context.Background())
			done := make(chan error, 1)
			go func() { done <- runner.Run(ctx) }()
			for _, b := range tt.barriers {
				if err := runner.PushBarrier(context.Background(), b); err != nil {
					t.Fatalf("push barrier: %v", err)
				}
			}
			var got error
			select {
			case got = <-done:
			case <-time.After(time.Second):
				t.Fatal("expected barrier failure")
			}
			cancel()
			if !errors.Is(got, tt.want) {
				t.Fatalf("error = %v, want %v", got, tt.want)
			}
			classified := kernel.Classify(got)
			if classified.Category != kernel.CategoryProtocol {
				t.Fatalf("category = %s, want protocol_violation", classified.Category)
			}
		})
	}
}

func TestConcurrentCheckpointRangeIsExplicitlyRejected(t *testing.T) {
	cfg := testConfig(protocol.Aligned)
	cfg.MaxConcurrentCheckpoints = 2
	_, err := kernel.NewRunner(cfg, nil)
	if !errors.Is(err, kernel.ErrConcurrentCheckpoint) {
		t.Fatalf("NewRunner error = %v, want concurrent checkpoint rejection", err)
	}
}
