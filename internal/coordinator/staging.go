package coordinator

import (
	"sync"

	"opp300-speculative-batch/internal/artifacts"
)

type stagingRegistry struct {
	mu     sync.Mutex
	values map[string]artifacts.StagedArtifact
}

func newStagingRegistry() *stagingRegistry {
	return &stagingRegistry{values: map[string]artifacts.StagedArtifact{}}
}

func (r *stagingRegistry) Put(staged artifacts.StagedArtifact) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.values[staged.ID] = staged
}

func (r *stagingRegistry) Get(id string) (artifacts.StagedArtifact, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	staged, ok := r.values[id]
	return staged, ok
}

func (r *stagingRegistry) Delete(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.values, id)
}
