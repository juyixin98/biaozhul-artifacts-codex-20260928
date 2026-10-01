package kernel

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"sync"

	"checkpointlab/internal/protocol"
	"checkpointlab/internal/storage"
)

type Config struct {
	JobID         string        `json:"job_id"`
	Mode          protocol.Mode `json:"mode"`
	GateCapacity  int           `json:"gate_capacity"`
	InflightLimit int           `json:"inflight_limit"`
}

type Metrics struct {
	ProcessedRecords  int64                `json:"processed_records"`
	SealedCheckpoints int64                `json:"sealed_checkpoints"`
	RejectedEvents    int64                `json:"rejected_events"`
	PeakInflight      int                  `json:"peak_inflight"`
	Inflight          int                  `json:"inflight"`
	CommittedEpoch    int64                `json:"committed_epoch"`
	ActiveEpoch       int64                `json:"active_epoch"`
	Gates             map[string]GateStats `json:"gates"`
}

type Result struct {
	Accepted      bool          `json:"accepted"`
	Stream        string        `json:"stream"`
	Offset        int64         `json:"offset"`
	Epoch         int64         `json:"epoch"`
	Committed     bool          `json:"committed"`
	ActiveEpoch   int64         `json:"active_epoch"`
	InflightDelta int           `json:"inflight_delta"`
	OutputID      string        `json:"output_id,omitempty"`
	StateVersion  int64         `json:"state_version"`
	State         storage.State `json:"state"`
	Metrics       Metrics       `json:"metrics"`
}

type activeCheckpoint struct {
	epoch       int64
	attemptID   string
	base        storage.State
	baseOffsets map[string]int64
	targets     map[string]int64
	records     map[string][]storage.RecordRef
}

type Operator struct {
	cfg   Config
	store storage.Store
	gates map[string]*Gate

	mu             sync.Mutex
	state          storage.State
	next           map[string]int64
	lastBarrier    map[string]int64
	active         *activeCheckpoint
	committedEpoch int64
	stateVersion   int64
	processed      int64
	sealed         int64
	rejected       int64
	peakInflight   int
	fatal          error
}

func NewOperator(cfg Config, store storage.Store) (*Operator, error) {
	if cfg.JobID == "" {
		return nil, protocol.NewTypedError(protocol.CategoryValidation, "job_id is required", false)
	}
	if !cfg.Mode.Valid() {
		return nil, protocol.NewTypedError(protocol.CategoryValidation, "mode must be aligned or unaligned", false)
	}
	if cfg.GateCapacity < 1 {
		return nil, protocol.NewTypedError(protocol.CategoryValidation, "gate_capacity must be positive", false)
	}
	if cfg.Mode == protocol.ModeUnaligned && cfg.InflightLimit < 1 {
		return nil, protocol.NewTypedError(protocol.CategoryValidation, "inflight_limit must be positive in unaligned mode", false)
	}
	return &Operator{
		cfg:   cfg,
		store: store,
		gates: map[string]*Gate{
			protocol.StreamA: NewGate(protocol.StreamA, cfg.GateCapacity),
			protocol.StreamB: NewGate(protocol.StreamB, cfg.GateCapacity),
		},
		state:       storage.State{},
		next:        map[string]int64{protocol.StreamA: 0, protocol.StreamB: 0},
		lastBarrier: map[string]int64{protocol.StreamA: 0, protocol.StreamB: 0},
	}, nil
}

func (o *Operator) Gate(stream string) *Gate { return o.gates[stream] }

func (o *Operator) SnapshotState() storage.State {
	o.mu.Lock()
	defer o.mu.Unlock()
	return cloneState(o.state)
}

func (o *Operator) Restore(ctx context.Context) (storage.RestorePoint, error) {
	point, err := o.store.Restore(ctx, o.cfg.JobID)
	if err != nil {
		return point, err
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.state = storage.State{}
	o.active = nil
	o.committedEpoch = point.CommittedEpoch
	o.next = map[string]int64{protocol.StreamA: 0, protocol.StreamB: 0}
	o.lastBarrier = map[string]int64{protocol.StreamA: 0, protocol.StreamB: 0}
	if point.CommittedEpoch > 0 {
		o.state = cloneState(point.Snapshot.Base)
		for _, stream := range []string{protocol.StreamA, protocol.StreamB} {
			o.next[stream] = point.Snapshot.Targets[stream].Offset
			o.lastBarrier[stream] = point.Snapshot.Epoch
		}
		streams := make([]string, 0, len(point.Snapshot.Inflights))
		for stream := range point.Snapshot.Inflights {
			streams = append(streams, stream)
		}
		sort.Strings(streams)
		for _, stream := range streams {
			refs := append([]storage.RecordRef(nil), point.Snapshot.Inflights[stream]...)
			sort.Slice(refs, func(i, j int) bool { return refs[i].Offset < refs[j].Offset })
			for _, ref := range refs {
				o.state[ref.Key] += ref.Value
			}
		}
		o.stateVersion++
	}
	return point, nil
}

func (o *Operator) Process(ctx context.Context, event protocol.Event, requestID string) (Result, error) {
	o.mu.Lock()
	if o.fatal != nil {
		err := o.fatal
		o.mu.Unlock()
		return Result{}, err
	}
	o.mu.Unlock()

	var result Result
	var err error
	switch event.Kind {
	case protocol.KindRecord:
		result, err = o.processRecord(ctx, event.Record, requestID)
	case protocol.KindBarrier:
		result, err = o.processBarrier(ctx, event.Barrier, requestID)
	default:
		err = protocol.NewTypedError(protocol.CategoryValidation, "unknown event kind", false)
	}
	if err != nil {
		o.mu.Lock()
		o.rejected++
		var typed *protocol.TypedError
		if !errors.As(err, &typed) {
			o.fatal = err
		}
		o.mu.Unlock()
		_ = o.log(ctx, requestID, 0, "event_rejected", "kernel", err.Error())
	}
	return result, err
}

func (o *Operator) Metrics() Metrics {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.metricsLocked()
}

func (o *Operator) Close() {
	o.gates[protocol.StreamA].Close()
	o.gates[protocol.StreamB].Close()
}

func (o *Operator) metricsLocked() Metrics {
	return Metrics{
		ProcessedRecords:  o.processed,
		SealedCheckpoints: o.sealed,
		RejectedEvents:    o.rejected,
		PeakInflight:      o.peakInflight,
		Inflight:          o.inflightCountLocked(),
		CommittedEpoch:    o.committedEpoch,
		ActiveEpoch:       o.activeEpochLocked(),
		Gates: map[string]GateStats{
			protocol.StreamA: o.gates[protocol.StreamA].Stats(),
			protocol.StreamB: o.gates[protocol.StreamB].Stats(),
		},
	}
}

func (o *Operator) activeEpochLocked() int64 {
	if o.active == nil {
		return 0
	}
	return o.active.epoch
}

func (o *Operator) inflightCountLocked() int {
	if o.active == nil {
		return 0
	}
	return len(o.active.records[protocol.StreamA]) + len(o.active.records[protocol.StreamB])
}

func cloneState(in storage.State) storage.State {
	out := make(storage.State, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func outputID(jobID, stream string, offset int64) string {
	return fmt.Sprintf("%s:%s:%020d", jobID, stream, offset)
}

func newAttemptID() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

func isTyped(err error) bool {
	var typed *protocol.TypedError
	return errors.As(err, &typed)
}

func (o *Operator) log(ctx context.Context, requestID string, epoch int64, step, location, detail string) error {
	return o.store.AppendEvent(ctx, storage.EventLogEntry{
		JobID:     o.cfg.JobID,
		RequestID: requestID,
		Epoch:     epoch,
		Step:      step,
		Location:  location,
		Detail:    detail,
	})
}
