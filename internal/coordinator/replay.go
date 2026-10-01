package coordinator

import (
	"encoding/json"
	"net/http"
	"strconv"

	"opp300-speculative-batch/internal/protocol"
)

type replayRequest struct {
	IDempotencyKey string          `json:"idempotency_key"`
	Policy         protocol.Policy `json:"policy"`
}

func (s *Server) replayEvents(w http.ResponseWriter, r *http.Request) {
	sourceID := r.PathValue("id")
	events, err := s.Store.ListEvents(r.Context(), sourceID, 0, 1000)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	var request replayRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "invalid JSON: "+err.Error())
		return
	}
	shards, err := s.Store.ListShards(r.Context(), sourceID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if len(shards) == 0 {
		writeError(w, http.StatusConflict, "REPLAY_EMPTY", "source job has no shard inputs")
		return
	}

	inputs := make([]protocol.ShardInput, 0, len(shards))
	for _, shard := range shards {
		inputs = append(inputs, protocol.ShardInput{
			Ordinal:  shard.Ordinal,
			Key:      shard.Key,
			Input:    shard.Input,
			Behavior: shard.Behavior,
		})
	}
	idempotencyKey := request.IDempotencyKey
	if idempotencyKey == "" {
		idempotencyKey = "replay-" + sourceID
	}
	policy := request.Policy
	if policy.LeaseMS == 0 && policy.SpeculateAfterMS == 0 {
		policy, err = s.Store.GetPolicy(r.Context(), sourceID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
	}
	created, err := s.Store.CreateJob(r.Context(), protocol.JobSpec{
		IDempotencyKey: idempotencyKey,
		Shards:         inputs,
		Policy:         policy,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}

	replayPayload, _ := json.Marshal(map[string]any{
		"source_job_id": sourceID,
		"event_count":   len(events),
		"reason":        "replay uses immutable inputs and creates a new run; original events are not rewritten",
	})
	response := map[string]any{
		"job":            created.Job,
		"shards":         created.Shards,
		"already_exists": created.AlreadyExists,
		"source_events":  events,
		"replay_basis":   json.RawMessage(replayPayload),
	}
	status := http.StatusCreated
	if created.AlreadyExists {
		status = http.StatusOK
	}
	writeJSON(w, status, response)
}

func parsePositiveInt(raw string) (int, bool) {
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return 0, false
	}
	return value, true
}
