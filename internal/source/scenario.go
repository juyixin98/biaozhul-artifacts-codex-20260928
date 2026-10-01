package source

import (
	"encoding/json"
	"fmt"
	"os"

	"checkpointlab/internal/protocol"
)

type Scenario struct {
	Name           string         `json:"name"`
	Description    string         `json:"description"`
	Mode           protocol.Mode  `json:"mode"`
	GateCapacity   int            `json:"gate_capacity"`
	InflightLimit  int            `json:"inflight_limit"`
	Steps          []Step         `json:"steps"`
	ExpectedReport ExpectedReport `json:"expected_report"`
}

type Step struct {
	Name        string                 `json:"name"`
	Stream      string                 `json:"stream,omitempty"`
	Kind        protocol.EventKind     `json:"kind,omitempty"`
	Offset      int64                  `json:"offset,omitempty"`
	Epoch       int64                  `json:"epoch,omitempty"`
	Key         string                 `json:"key,omitempty"`
	Value       int64                  `json:"value,omitempty"`
	ExpectError protocol.ErrorCategory `json:"expect_error,omitempty"`
}

type ExpectedReport struct {
	CommittedEpoch int64            `json:"committed_epoch"`
	CommittedCount map[string]int64 `json:"committed_count"`
	InflightPeak   int              `json:"inflight_peak"`
	BlockedA       int64            `json:"blocked_a"`
	BlockedB       int64            `json:"blocked_b"`
	ReplayRecords  int              `json:"replay_records"`
}

func LoadFile(path string) (Scenario, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Scenario{}, err
	}
	return Load(b)
}

func Load(b []byte) (Scenario, error) {
	var sc Scenario
	if err := json.Unmarshal(b, &sc); err != nil {
		return Scenario{}, err
	}
	if err := sc.Validate(); err != nil {
		return Scenario{}, err
	}
	return sc, nil
}

func (s Scenario) Validate() error {
	if s.Name == "" {
		return fmt.Errorf("scenario name is required")
	}
	if !s.Mode.Valid() {
		return fmt.Errorf("scenario mode must be aligned or unaligned")
	}
	if s.GateCapacity < 1 {
		return fmt.Errorf("gate_capacity must be positive")
	}
	if s.Mode == protocol.ModeUnaligned && s.InflightLimit < 1 {
		return fmt.Errorf("inflight_limit must be positive")
	}
	next := map[string]int64{protocol.StreamA: 0, protocol.StreamB: 0}
	for i, step := range s.Steps {
		if !protocol.ValidStream(step.Stream) {
			return fmt.Errorf("step %d has invalid stream", i)
		}
		if step.Kind == protocol.KindRecord {
			if step.Offset != next[step.Stream] {
				return fmt.Errorf("step %d has offset %d, expected %d", i, step.Offset, next[step.Stream])
			}
			if step.Key == "" {
				return fmt.Errorf("step %d key is required", i)
			}
			next[step.Stream] = step.Offset + 1
		}
	}
	return nil
}

func (s Step) Event() protocol.Event {
	if s.Kind == protocol.KindBarrier {
		return protocol.BarrierEvent(protocol.Barrier{
			Stream: s.Stream,
			Epoch:  s.Epoch,
			Offset: s.Offset,
		})
	}
	return protocol.RecordEvent(protocol.Record{
		Stream: s.Stream,
		Offset: s.Offset,
		Epoch:  s.Epoch,
		Key:    s.Key,
		Value:  s.Value,
	})
}

func (s Scenario) RecordsAtOrAfter(start map[string]int64) []Step {
	var result []Step
	for _, step := range s.Steps {
		if step.Kind == protocol.KindRecord && step.Offset >= start[step.Stream] {
			result = append(result, step)
		}
	}
	return result
}
