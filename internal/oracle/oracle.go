package oracle

import (
	"checkpointlab/internal/protocol"
	"checkpointlab/internal/source"
)

type Expected struct {
	CommittedEpoch int64            `json:"committed_epoch"`
	CommittedState map[string]int64 `json:"committed_state"`
	CommittedCount map[string]int64 `json:"committed_count"`
	TotalState     map[string]int64 `json:"total_state"`
	InflightPeak   int              `json:"inflight_peak"`
	ReplayRecords  int              `json:"replay_records"`
}

// ExpectedBoundary is independent of internal/kernel. It is a direct reference
// calculation over the synthetic script: committed records are all records
// below both barrier offsets; in unaligned mode records between the first
// barrier arrival position and the slow barrier are replay records.
func ExpectedBoundary(sc source.Scenario) Expected {
	next := map[string]int64{"A": 0, "B": 0}
	baseOffsets := map[string]int64{"A": 0, "B": 0}
	targets := map[string]int64{}
	firstBarrierSeen := false
	var epoch int64

	for _, step := range sc.Steps {
		if step.Kind == protocol.KindBarrier {
			if !firstBarrierSeen {
				baseOffsets["A"] = next["A"]
				baseOffsets["B"] = next["B"]
				firstBarrierSeen = true
			}
			targets[step.Stream] = step.Offset
			if step.Epoch > epoch {
				epoch = step.Epoch
			}
		}
		if step.Kind == protocol.KindRecord {
			next[step.Stream] = step.Offset + 1
		}
	}

	committedState := map[string]int64{}
	committedCount := map[string]int64{}
	totalState := map[string]int64{}
	replay := 0
	for _, step := range sc.Steps {
		if step.Kind != protocol.KindRecord {
			continue
		}
		totalState[step.Key] += step.Value
		if target, ok := targets[step.Stream]; ok && step.Offset < target {
			committedState[step.Key] += step.Value
			committedCount[step.Key]++
			if sc.Mode == protocol.ModeUnaligned && step.Offset >= baseOffsets[step.Stream] {
				replay++
			}
		}
	}
	peak := 0
	if sc.Mode == protocol.ModeUnaligned {
		for stream := range targets {
			peak += int(targets[stream] - baseOffsets[stream])
		}
	}

	return Expected{
		CommittedEpoch: epoch,
		CommittedState: committedState,
		CommittedCount: committedCount,
		TotalState:     totalState,
		InflightPeak:   peak,
		ReplayRecords:  replay,
	}
}
