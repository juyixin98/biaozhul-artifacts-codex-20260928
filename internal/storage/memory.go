package storage

import (
	"context"
	"sort"
	"sync"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
)

type MemoryStore struct {
	mu sync.Mutex
	// cps stores only complete checkpoints. Failed injections stay absent,
	// matching the recovery rule "roll back to the newest complete snapshot".
	cps map[string][]protocol.Checkpoint
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{cps: map[string][]protocol.Checkpoint{}}
}

func (s *MemoryStore) Init(context.Context) error { return nil }

func (s *MemoryStore) SaveCheckpoint(_ context.Context, cp protocol.Checkpoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, existing := range s.cps[cp.TaskID] {
		if existing.ID == cp.ID {
			if checkpointsEqual(existing, cp) {
				return nil
			}
			return &kernel.ClassifiedError{Category: kernel.CategoryStorage, Err: kernel.ErrDuplicateBarrier}
		}
		if existing.ID > cp.ID {
			return &kernel.ClassifiedError{Category: kernel.CategoryProtocol, Err: kernel.ErrBarrierOutOfOrder}
		}
	}
	s.cps[cp.TaskID] = append(s.cps[cp.TaskID], cp)
	sort.Slice(s.cps[cp.TaskID], func(i, j int) bool {
		return s.cps[cp.TaskID][i].ID < s.cps[cp.TaskID][j].ID
	})
	return nil
}

func (s *MemoryStore) GetLatest(_ context.Context, taskID string) (protocol.Checkpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	items := s.cps[taskID]
	if len(items) == 0 {
		return protocol.Checkpoint{}, &kernel.ClassifiedError{Category: kernel.CategoryNotFound, Err: kernel.ErrCheckpointNotFound}
	}
	return cloneCheckpoint(items[len(items)-1]), nil
}

func (s *MemoryStore) GetCheckpoint(_ context.Context, taskID string, id int64) (protocol.Checkpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, cp := range s.cps[taskID] {
		if cp.ID == id {
			return cloneCheckpoint(cp), nil
		}
	}
	return protocol.Checkpoint{}, &kernel.ClassifiedError{Category: kernel.CategoryNotFound, Err: kernel.ErrCheckpointNotFound}
}

func (s *MemoryStore) ListCheckpoints(_ context.Context, taskID string, limit int) ([]protocol.Checkpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	items := s.cps[taskID]
	if limit <= 0 || limit > len(items) {
		limit = len(items)
	}
	page := items[len(items)-limit:]
	out := make([]protocol.Checkpoint, 0, len(page))
	for i := len(page) - 1; i >= 0; i-- {
		out = append(out, cloneCheckpoint(page[i]))
	}
	return out, nil
}
