package policy

import (
	"time"
)

const ProtocolVersion = "specbatch/v1"

type Progress struct {
	Elapsed          time.Duration
	Reported         float64
	SpecAfter        time.Duration
	SpecAtProgress   float64
	AttemptActive    bool
	ActiveAttempts   int
	MaxConcurrent    int
	UsedSpeculative  int
	TotalBudget      int
	TerminalFailures int
	TotalAttempts    int
}

type Decision struct {
	Launch  bool
	Reasons []string
}

// ShouldSpeculate is intentionally explicit: a backup needs a slow/stuck
// observation and an unused per-shard concurrency plus job-wide copy budget.
func ShouldSpeculate(p Progress) Decision {
	if !p.AttemptActive || p.ActiveAttempts >= p.MaxConcurrent {
		return Decision{Reasons: []string{"no available speculative slot"}}
	}
	if p.UsedSpeculative >= p.TotalBudget {
		return Decision{Reasons: []string{"job speculative copy budget exhausted"}}
	}
	slowByTime := p.Elapsed >= p.SpecAfter && p.SpecAfter > 0
	slowByProgress := p.Reported <= p.SpecAtProgress && p.SpecAtProgress > 0
	if !slowByTime && !slowByProgress {
		return Decision{Reasons: []string{"progress threshold not reached"}}
	}
	reasons := []string{}
	if slowByTime {
		reasons = append(reasons, "elapsed threshold reached without completion")
	}
	if slowByProgress {
		reasons = append(reasons, "reported progress below configured checkpoint")
	}
	return Decision{Launch: true, Reasons: reasons}
}
