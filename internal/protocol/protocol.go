package protocol

import "fmt"

type Mode string

const (
	Aligned   Mode = "aligned"
	Unaligned Mode = "unaligned"
)

const (
	StreamA = "A"
	StreamB = "B"
)

func (m Mode) Valid() bool {
	return m == Aligned || m == Unaligned
}

// Position is the position that a stream has handed to the operator. Positions
// are dense synthetic offsets: after restart the replay source first emits
// inflight records and then resumes at BarrierAfter+1.
type Position struct {
	Stream       string `json:"stream"`
	Seq          int64  `json:"seq"`
	BarrierAfter int64  `json:"barrier_after"`
}

type Record struct {
	Stream string `json:"stream"`
	Key    string `json:"key"`
	Value  int64  `json:"value"`
	Seq    int64  `json:"seq"`
}

func (r Record) ID() string {
	return r.Stream + fmt.Sprintf("-%020d", r.Seq)
}

type Barrier struct {
	ID     int64  `json:"id"`
	Stream string `json:"stream"`
	After  int64  `json:"after"`
	// OutOfOrder marks a deliberately replayed or late barrier for tests.
	OutOfOrder bool `json:"out_of_order,omitempty"`
}

type PairCount struct {
	A int64 `json:"a"`
	B int64 `json:"b"`
}

type State struct {
	Counts map[string]PairCount `json:"counts"`
}

func NewState() *State {
	return &State{Counts: map[string]PairCount{}}
}

func (s *State) Clone() *State {
	out := NewState()
	for key, count := range s.Counts {
		out.Counts[key] = count
	}
	return out
}

// Apply updates the reference keyed state and returns the resulting concrete
// output. The teaching operator deliberately keeps this transformation small
// so tests can independently calculate expected A/B reference counts.
func (s *State) Apply(r Record) Output {
	count := s.Counts[r.Key]
	switch r.Stream {
	case StreamA:
		count.A += r.Value
	case StreamB:
		count.B += r.Value
	default:
		panic(fmt.Sprintf("unknown stream %q", r.Stream))
	}
	s.Counts[r.Key] = count
	return Output{
		ID:     fmt.Sprintf("%s-%s-%020d", r.Stream, r.Key, r.Seq),
		Key:    r.Key,
		Seq:    r.Seq,
		Stream: r.Stream,
		Count:  count,
	}
}

type Output struct {
	ID     string    `json:"id"`
	Stream string    `json:"stream"`
	Key    string    `json:"key"`
	Seq    int64     `json:"seq"`
	Count  PairCount `json:"count"`
	Epoch  int64     `json:"epoch"`
}

type InFlight struct {
	Record         Record `json:"record"`
	ObservedBefore bool   `json:"observed_before"`
}

type Checkpoint struct {
	TaskID     string                `json:"task_id"`
	ID         int64                 `json:"id"`
	Mode       Mode                  `json:"mode"`
	State      *State                `json:"state"`
	InFlight   map[string][]InFlight `json:"inflight"`
	Positions  map[string]Position   `json:"positions"`
	Outputs    []Output              `json:"outputs"`
	Metrics    Metrics               `json:"metrics"`
	StartedAt  string                `json:"started_at"`
	FinishedAt string                `json:"finished_at"`
	Status     CheckpointStatus      `json:"status"`
	Failure    *Failure              `json:"failure,omitempty"`
}

type CheckpointStatus string

const (
	CheckpointPending  CheckpointStatus = "pending"
	CheckpointComplete CheckpointStatus = "complete"
	CheckpointFailed   CheckpointStatus = "failed"
)

type Failure struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

type Config struct {
	TaskID                   string `json:"task_id"`
	Mode                     Mode   `json:"mode"`
	BackpressureLimit        int    `json:"backpressure_limit"`
	MaxConcurrentCheckpoints int    `json:"max_concurrent_checkpoints"`
	FailCheckpointID         int64  `json:"fail_checkpoint_id,omitempty"`
}

func (c Config) Validate() error {
	if c.TaskID == "" {
		return fmt.Errorf("task_id is required")
	}
	if !c.Mode.Valid() {
		return fmt.Errorf("mode must be %q or %q", Aligned, Unaligned)
	}
	if c.BackpressureLimit <= 0 {
		return fmt.Errorf("backpressure_limit must be positive")
	}
	if c.MaxConcurrentCheckpoints <= 0 {
		return fmt.Errorf("max_concurrent_checkpoints must be positive")
	}
	return nil
}

type Metrics struct {
	Processed        map[string]int64 `json:"processed"`
	Replayed         int64            `json:"replayed"`
	BufferOccupancy  int              `json:"buffer_occupancy"`
	MaxBuffer        int              `json:"max_buffer"`
	CommittedOutputs int              `json:"committed_outputs"`
	PendingOutputs   int              `json:"pending_outputs"`
}
