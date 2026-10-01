package lab_test

import (
	"context"
	"errors"
	"testing"

	"checkpointlab/internal/kernel"
	"checkpointlab/internal/lab"
	"checkpointlab/internal/protocol"
	"checkpointlab/internal/source"
	"checkpointlab/internal/storage"
)

func TestBothModeReportsAgainstIndependentOracle(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name     string
		path     string
		inflight int
		blocked  int64
	}{
		{name: "aligned", path: "../../testdata/scenarios/aligned-speed-diff.json", inflight: 0, blocked: 1},
		{name: "unaligned", path: "../../testdata/scenarios/unaligned-speed-diff.json", inflight: 1, blocked: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sc, err := source.LoadFile(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			store := storage.NewMemoryStore()
			runner := lab.NewRunner(store)
			op, report, err := runner.Run(ctx, "compare-"+tc.name, sc, "req-"+tc.name)
			if err != nil {
				t.Fatal(err)
			}
			defer op.Close()
			if len(report.Failures) != 0 {
				t.Fatalf("reference failures: %+v", report.Failures)
			}
			if report.CommittedState["x"] != 5 || report.CommittedState["y"] != 2 {
				t.Fatalf("committed=%v", report.CommittedState)
			}
			if report.Metrics.PeakInflight != tc.inflight {
				t.Fatalf("peak inflight=%d want %d", report.Metrics.PeakInflight, tc.inflight)
			}
			if report.Metrics.Gates["A"].BlockedAttempts != tc.blocked {
				t.Fatalf("blocked A=%d want %d", report.Metrics.Gates["A"].BlockedAttempts, tc.blocked)
			}
		})
	}
}

func TestSnapshotInterruptionLeavesNoCommittedBoundaryAndReplaysFromStart(t *testing.T) {
	sc, err := source.LoadFile("../../testdata/scenarios/unaligned-speed-diff.json")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	base := storage.NewMemoryStore()
	store := &faultStore{Store: base, failObserve: true}
	if err := store.CreateJob(ctx, "interrupted", sc.Mode, nil); err != nil {
		t.Fatal(err)
	}
	op, err := kernel.NewOperator(kernel.Config{
		JobID: "interrupted", Mode: sc.Mode, GateCapacity: 3, InflightLimit: 10,
	}, store)
	if err != nil {
		t.Fatal(err)
	}

	for i, step := range sc.Steps {
		_, err := op.Process(ctx, step.Event(), "req-interrupt")
		if i == len(sc.Steps)-1 {
			var typed *protocol.TypedError
			if !errors.As(err, &typed) || typed.Category != protocol.CategoryStorageFailed {
				t.Fatalf("expected storage failure at second barrier, got %v", err)
			}
		} else if err != nil {
			t.Fatalf("%s: %v", step.Name, err)
		}
	}
	active, ok, err := base.ActiveCheckpoint(ctx, "interrupted")
	if err != nil || !ok || active.Status != storage.StatusActive {
		t.Fatalf("active=%+v ok=%v err=%v", active, ok, err)
	}

	newOp, err := kernel.NewOperator(kernel.Config{
		JobID: "interrupted", Mode: sc.Mode, GateCapacity: 3, InflightLimit: 10,
	}, base)
	if err != nil {
		t.Fatal(err)
	}
	point, err := newOp.Restore(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if point.CommittedEpoch != 0 {
		t.Fatalf("interrupted snapshot must not be committed, got epoch %d", point.CommittedEpoch)
	}
	if got := len(newOp.SnapshotState()); got != 0 {
		t.Fatalf("cold restore state entries=%d", got)
	}
	outs, _ := base.ListOutputs(ctx, "interrupted", false)
	if len(outs) != 0 {
		t.Fatalf("speculative outputs after interrupted restore=%d", len(outs))
	}
}

type faultStore struct {
	storage.Store
	failObserve bool
}

func (f *faultStore) ObserveBarrier(ctx context.Context, jobID string, epoch int64, attemptID string, b protocol.Barrier) error {
	if f.failObserve {
		return protocol.NewTypedError(protocol.CategoryStorageFailed, "injected snapshot interruption", false)
	}
	return f.Store.ObserveBarrier(ctx, jobID, epoch, attemptID, b)
}
