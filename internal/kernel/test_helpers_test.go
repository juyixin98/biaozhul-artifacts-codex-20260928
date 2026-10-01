package kernel_test

import (
	"context"
	"sync"

	"opp299/internal/protocol"
)

type recordingStore struct {
	mu          sync.Mutex
	checkpoints []protocol.Checkpoint
}

func newRecordingStore() *recordingStore {
	return &recordingStore{}
}

func (s *recordingStore) SaveCheckpoint(_ context.Context, cp protocol.Checkpoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.checkpoints = append(s.checkpoints, cp)
	return nil
}

func (s *recordingStore) get(index int) protocol.Checkpoint {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.checkpoints[index]
}

func (s *recordingStore) len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.checkpoints)
}
