package replay

import (
	"fmt"

	"checkpointlab/internal/source"
	"checkpointlab/internal/storage"
)

type Plan struct {
	JobID          string           `json:"job_id"`
	CommittedEpoch int64            `json:"committed_epoch"`
	ReplayStart    map[string]int64 `json:"replay_start"`
	Steps          []source.Step    `json:"steps"`
	Reason         string           `json:"reason"`
}

type Planner struct {
	scenario source.Scenario
}

func NewPlanner(scenario source.Scenario) *Planner {
	return &Planner{scenario: scenario}
}

func (p *Planner) Build(point storage.RestorePoint) Plan {
	start := point.ReplayStart
	if len(start) == 0 {
		start = map[string]int64{"A": 0, "B": 0}
	}
	steps := p.scenario.RecordsAtOrAfter(start)
	reason := "cold start: replay every synthetic record"
	if point.CommittedEpoch > 0 {
		reason = fmt.Sprintf("restored epoch %d; replay only offsets at or after durable barrier targets", point.CommittedEpoch)
	}
	return Plan{
		JobID:          point.Snapshot.JobID,
		CommittedEpoch: point.CommittedEpoch,
		ReplayStart:    start,
		Steps:          steps,
		Reason:         reason,
	}
}
