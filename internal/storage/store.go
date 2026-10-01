package storage

import (
	"context"
	"time"

	"checkpointlab/internal/protocol"
)

type Status string

const (
	StatusActive     Status = "ACTIVE"
	StatusSealed     Status = "SEALED"
	StatusSuperseded Status = "SUPERSEDED"
)

type BarrierTarget struct {
	Offset int64 `json:"offset"`
	Seen   bool  `json:"seen"`
}

type State map[string]int64

// Snapshot is the durable checkpoint boundary. Inflights contains exactly the
// records before the slower barrier that were not in Base when the first
// unaligned barrier arrived. In aligned mode Inflights is always empty.
type Snapshot struct {
	JobID        string                   `json:"job_id"`
	Epoch        int64                    `json:"epoch"`
	AttemptID    string                   `json:"attempt_id"`
	Mode         protocol.Mode            `json:"mode"`
	Status       Status                   `json:"status"`
	Base         State                    `json:"base"`
	Inflights    map[string][]RecordRef   `json:"inflights"`
	Targets      map[string]BarrierTarget `json:"targets"`
	SeenBarriers map[string]bool          `json:"seen_barriers"`
	BaseOffsets  map[string]int64         `json:"base_offsets"`
	Version      int64                    `json:"version"`
	CreatedAt    time.Time                `json:"created_at"`
}

type RecordRef struct {
	Stream string `json:"stream"`
	Offset int64  `json:"offset"`
	Epoch  int64  `json:"epoch"`
	Key    string `json:"key"`
	Value  int64  `json:"value"`
}

type Output struct {
	JobID     string `json:"job_id"`
	Epoch     int64  `json:"epoch"`
	Stream    string `json:"stream"`
	Offset    int64  `json:"offset"`
	Key       string `json:"key"`
	Value     int64  `json:"value"`
	Committed bool   `json:"committed"`
	OutputID  string `json:"output_id"`
}

type RestorePoint struct {
	Snapshot       Snapshot         `json:"snapshot"`
	ReplayStart    map[string]int64 `json:"replay_start"`
	CommittedEpoch int64            `json:"committed_epoch"`
}

type EventLogEntry struct {
	JobID     string `json:"job_id"`
	RequestID string `json:"request_id"`
	Epoch     int64  `json:"epoch"`
	Step      string `json:"step"`
	Location  string `json:"location"`
	Detail    string `json:"detail"`
}

type Store interface {
	CreateJob(ctx context.Context, jobID string, mode protocol.Mode, configJSON []byte) error
	JobMode(ctx context.Context, jobID string) (protocol.Mode, error)
	StartCheckpoint(ctx context.Context, snap Snapshot) error
	AppendInflight(ctx context.Context, jobID string, epoch int64, attemptID string, ref RecordRef) error
	ObserveBarrier(ctx context.Context, jobID string, epoch int64, attemptID string, b protocol.Barrier) error
	// SealCheckpoint publishes the boundary and atomically commits every
	// staged output with Epoch <= epoch. A caller must treat an error as an
	// interrupted snapshot unless the store later proves the row is sealed.
	SealCheckpoint(ctx context.Context, jobID string, epoch int64, attemptID string, base State, records []RecordRef) (Snapshot, error)
	GetCheckpoint(ctx context.Context, jobID string, epoch int64) (Snapshot, bool, error)
	LatestSealed(ctx context.Context, jobID string) (Snapshot, bool, error)
	ActiveCheckpoint(ctx context.Context, jobID string) (Snapshot, bool, error)
	StageOutput(ctx context.Context, out Output) error
	ListOutputs(ctx context.Context, jobID string, committedOnly bool) ([]Output, error)
	Restore(ctx context.Context, jobID string) (RestorePoint, error)
	AppendEvent(ctx context.Context, entry EventLogEntry) error
	ListEvents(ctx context.Context, jobID string, limit int) ([]EventLogEntry, error)
	Close() error
}

func RefFromRecord(r protocol.Record) RecordRef {
	return RecordRef{Stream: r.Stream, Offset: r.Offset, Epoch: r.Epoch, Key: r.Key, Value: r.Value}
}

func RefRecord(r RecordRef) protocol.Record {
	return protocol.Record{Stream: r.Stream, Offset: r.Offset, Epoch: r.Epoch, Key: r.Key, Value: r.Value}
}
