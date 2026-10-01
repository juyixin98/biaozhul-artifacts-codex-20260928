package storage

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"checkpointlab/internal/protocol"
)

type MemoryStore struct {
	mu          sync.Mutex
	jobs        map[string]protocol.Mode
	checkpoints map[string]map[int64]Snapshot
	records     map[string]map[string]RecordRef
	outputs     map[string][]Output
	events      map[string][]EventLogEntry
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		jobs:        make(map[string]protocol.Mode),
		checkpoints: make(map[string]map[int64]Snapshot),
		records:     make(map[string]map[string]RecordRef),
		outputs:     make(map[string][]Output),
		events:      make(map[string][]EventLogEntry),
	}
}

func (s *MemoryStore) CreateJob(_ context.Context, jobID string, mode protocol.Mode, _ []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.jobs[jobID]; ok {
		return fmt.Errorf("job %s already exists", jobID)
	}
	s.jobs[jobID] = mode
	s.checkpoints[jobID] = make(map[int64]Snapshot)
	s.records[jobID] = make(map[string]RecordRef)
	return nil
}

func (s *MemoryStore) JobMode(_ context.Context, jobID string) (protocol.Mode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	mode, ok := s.jobs[jobID]
	if !ok {
		return "", fmt.Errorf("job %s not found", jobID)
	}
	return mode, nil
}

func (s *MemoryStore) StartCheckpoint(_ context.Context, snap Snapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.requireJob(snap.JobID); err != nil {
		return err
	}
	for _, existing := range s.checkpoints[snap.JobID] {
		if existing.Status == StatusActive {
			return protocol.NewTypedError(protocol.CategoryCheckpointScope,
				fmt.Sprintf("job %s already has active checkpoint %d", snap.JobID, existing.Epoch), false)
		}
	}
	if _, ok := s.checkpoints[snap.JobID][snap.Epoch]; ok {
		return fmt.Errorf("checkpoint %d already exists", snap.Epoch)
	}
	snap.Status = StatusActive
	snap.Version = 1
	s.checkpoints[snap.JobID][snap.Epoch] = cloneSnapshot(snap)
	return nil
}

func (s *MemoryStore) AppendInflight(_ context.Context, jobID string, epoch int64, _ string, ref RecordRef) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.requireJob(jobID); err != nil {
		return err
	}
	key := refKey(epoch, ref.Stream, ref.Offset)
	if _, exists := s.records[jobID][key]; exists {
		return nil
	}
	s.records[jobID][key] = ref
	snap := s.checkpoints[jobID][epoch]
	snap.Inflights[ref.Stream] = append(snap.Inflights[ref.Stream], ref)
	s.checkpoints[jobID][epoch] = snap
	return nil
}

func (s *MemoryStore) ObserveBarrier(_ context.Context, jobID string, epoch int64, attemptID string, b protocol.Barrier) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.checkpoints[jobID][epoch]
	if !ok {
		return fmt.Errorf("checkpoint %d not found", epoch)
	}
	if snap.AttemptID != attemptID || snap.Status != StatusActive {
		return protocol.NewTypedError(protocol.CategoryCheckpointScope, "checkpoint attempt is not active", false)
	}
	target := snap.Targets[b.Stream]
	if target.Seen {
		return protocol.NewTypedError(protocol.CategoryBarrierOrder,
			fmt.Sprintf("duplicate barrier stream=%s epoch=%d", b.Stream, epoch), false)
	}
	target.Seen = true
	target.Offset = b.Offset
	snap.Targets[b.Stream] = target
	snap.SeenBarriers[b.Stream] = true
	snap.Version++
	s.checkpoints[jobID][epoch] = snap
	return nil
}

func (s *MemoryStore) SealCheckpoint(_ context.Context, jobID string, epoch int64, attemptID string, base State, records []RecordRef) (Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.checkpoints[jobID][epoch]
	if !ok {
		return Snapshot{}, fmt.Errorf("checkpoint %d not found", epoch)
	}
	if snap.AttemptID != attemptID || snap.Status != StatusActive {
		return Snapshot{}, protocol.NewTypedError(protocol.CategoryCheckpointScope, "checkpoint attempt is not active", false)
	}
	for _, stream := range []string{protocol.StreamA, protocol.StreamB} {
		target := snap.Targets[stream]
		if !target.Seen {
			return Snapshot{}, protocol.NewTypedError(protocol.CategoryCheckpointScope,
				fmt.Sprintf("barrier for %s has not arrived", stream), false)
		}
	}
	snap.Base = cloneState(base)
	snap.Inflights = map[string][]RecordRef{}
	for _, ref := range records {
		snap.Inflights[ref.Stream] = append(snap.Inflights[ref.Stream], ref)
	}
	for stream := range snap.Inflights {
		sort.Slice(snap.Inflights[stream], func(i, j int) bool {
			return snap.Inflights[stream][i].Offset < snap.Inflights[stream][j].Offset
		})
	}
	snap.Status = StatusSealed
	snap.Version++
	s.checkpoints[jobID][epoch] = snap
	outs := s.outputs[jobID]
	for i := range outs {
		if outs[i].Epoch <= epoch {
			outs[i].Committed = true
		}
	}
	s.outputs[jobID] = outs
	return cloneSnapshot(snap), nil
}

func (s *MemoryStore) GetCheckpoint(_ context.Context, jobID string, epoch int64) (Snapshot, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.checkpoints[jobID][epoch]
	return cloneSnapshot(snap), ok, nil
}

func (s *MemoryStore) LatestSealed(_ context.Context, jobID string) (Snapshot, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.latestLocked(jobID, StatusSealed)
	return cloneSnapshot(snap), ok, nil
}

func (s *MemoryStore) ActiveCheckpoint(_ context.Context, jobID string) (Snapshot, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.latestLocked(jobID, StatusActive)
	return cloneSnapshot(snap), ok, nil
}

func (s *MemoryStore) StageOutput(_ context.Context, out Output) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.outputs[out.JobID] {
		if s.outputs[out.JobID][i].OutputID == out.OutputID {
			return nil
		}
	}
	s.outputs[out.JobID] = append(s.outputs[out.JobID], cloneOutput(out))
	return nil
}

func (s *MemoryStore) ListOutputs(_ context.Context, jobID string, committedOnly bool) ([]Output, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var result []Output
	for _, out := range s.outputs[jobID] {
		if !committedOnly || out.Committed {
			result = append(result, cloneOutput(out))
		}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Epoch != result[j].Epoch {
			return result[i].Epoch < result[j].Epoch
		}
		if result[i].Stream != result[j].Stream {
			return result[i].Stream < result[j].Stream
		}
		return result[i].Offset < result[j].Offset
	})
	return result, nil
}

func (s *MemoryStore) Restore(_ context.Context, jobID string) (RestorePoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	point := RestorePoint{ReplayStart: map[string]int64{protocol.StreamA: 0, protocol.StreamB: 0}}
	snap, ok := s.latestLocked(jobID, StatusSealed)
	if ok {
		for e, set := range s.checkpoints[jobID] {
			if set.Status == StatusActive && e >= snap.Epoch {
				set.Status = StatusSuperseded
				s.checkpoints[jobID][e] = set
			}
		}
		point.Snapshot = cloneSnapshot(snap)
		point.CommittedEpoch = snap.Epoch
		for _, stream := range []string{protocol.StreamA, protocol.StreamB} {
			point.ReplayStart[stream] = snap.Targets[stream].Offset
		}
		// Staged rows above the committed boundary are speculative work and
		// must be regenerated by deterministic replay.
		filtered := s.outputs[jobID][:0]
		for _, out := range s.outputs[jobID] {
			if out.Epoch <= snap.Epoch {
				out.Committed = true
				filtered = append(filtered, out)
			}
		}
		s.outputs[jobID] = append([]Output(nil), filtered...)
		return point, nil
	}
	for e, set := range s.checkpoints[jobID] {
		if set.Status == StatusActive {
			set.Status = StatusSuperseded
			s.checkpoints[jobID][e] = set
		}
	}
	s.outputs[jobID] = nil
	return point, nil
}

func (s *MemoryStore) AppendEvent(_ context.Context, entry EventLogEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events[entry.JobID] = append(s.events[entry.JobID], entry)
	return nil
}

func (s *MemoryStore) ListEvents(_ context.Context, jobID string, limit int) ([]EventLogEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries := append([]EventLogEntry(nil), s.events[jobID]...)
	if limit > 0 && len(entries) > limit {
		entries = entries[len(entries)-limit:]
	}
	return entries, nil
}

func (s *MemoryStore) Close() error { return nil }

func (s *MemoryStore) requireJob(jobID string) error {
	if _, ok := s.jobs[jobID]; !ok {
		return fmt.Errorf("job %s not found", jobID)
	}
	return nil
}

func (s *MemoryStore) latestLocked(jobID string, status Status) (Snapshot, bool) {
	var result Snapshot
	found := false
	for _, snap := range s.checkpoints[jobID] {
		if snap.Status == status && (!found || snap.Epoch > result.Epoch) {
			result, found = snap, true
		}
	}
	return result, found
}

func refKey(epoch int64, stream string, offset int64) string {
	return fmt.Sprintf("%d:%s:%d", epoch, stream, offset)
}

func cloneState(in State) State {
	out := make(State, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneOutput(in Output) Output { return in }

func cloneSnapshot(in Snapshot) Snapshot {
	out := in
	out.Base = cloneState(in.Base)
	out.Inflights = make(map[string][]RecordRef, len(in.Inflights))
	for stream, refs := range in.Inflights {
		out.Inflights[stream] = append([]RecordRef(nil), refs...)
	}
	out.Targets = make(map[string]BarrierTarget, len(in.Targets))
	for k, v := range in.Targets {
		out.Targets[k] = v
	}
	out.SeenBarriers = make(map[string]bool, len(in.SeenBarriers))
	for k, v := range in.SeenBarriers {
		out.SeenBarriers[k] = v
	}
	out.BaseOffsets = make(map[string]int64, len(in.BaseOffsets))
	for k, v := range in.BaseOffsets {
		out.BaseOffsets[k] = v
	}
	if out.Base == nil {
		out.Base = State{}
	}
	return out
}
