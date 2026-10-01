package policy

import (
	"testing"
	"time"
)

func TestSpeculationRequiresObservedSlowProgressAndBudget(t *testing.T) {
	base := Progress{
		Elapsed: 300 * time.Millisecond, Reported: 0.1, SpecAfter: 200 * time.Millisecond,
		SpecAtProgress: .25, AttemptActive: true, ActiveAttempts: 1, MaxConcurrent: 2,
		UsedSpeculative: 0, TotalBudget: 2,
	}
	if d := ShouldSpeculate(base); !d.Launch || len(d.Reasons) == 0 {
		t.Fatalf("expected explainable launch, got %+v", d)
	}
	noBudget := base
	noBudget.UsedSpeculative = 2
	if d := ShouldSpeculate(noBudget); d.Launch {
		t.Fatalf("budget exhausted must not launch: %+v", d)
	}
	fast := base
	fast.Elapsed = 10 * time.Millisecond
	fast.Reported = .9
	if d := ShouldSpeculate(fast); d.Launch {
		t.Fatalf("fast attempt must not launch duplicate: %+v", d)
	}
	noSlot := base
	noSlot.ActiveAttempts = 2
	if d := ShouldSpeculate(noSlot); d.Launch {
		t.Fatalf("full concurrency must not launch duplicate: %+v", d)
	}
}
