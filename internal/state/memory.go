package state

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"topk-demo/internal/protocol"
)

type MemoryStore struct {
	mu     sync.RWMutex
	runs   map[string]protocol.RunRecord
	events map[string][]protocol.EventRecord
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{runs: map[string]protocol.RunRecord{}, events: map[string][]protocol.EventRecord{}}
}

func (s *MemoryStore) SaveRun(_ context.Context, run protocol.RunRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.runs[run.RunID] = run
	return nil
}

func (s *MemoryStore) SaveEvent(_ context.Context, event protocol.EventRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events[event.RunID] = append(s.events[event.RunID], event)
	return nil
}

func (s *MemoryStore) GetRun(_ context.Context, runID string) (*protocol.RunRecord, []protocol.EventRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok {
		return nil, nil, fmt.Errorf("run %q not found", runID)
	}
	events := append([]protocol.EventRecord(nil), s.events[runID]...)
	sort.Slice(events, func(i, j int) bool { return events[i].Sequence < events[j].Sequence })
	return &run, events, nil
}

func (s *MemoryStore) Close() {}
