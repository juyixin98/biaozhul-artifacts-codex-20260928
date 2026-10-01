package coordinator

import (
	"errors"
	"net/http"

	"opp300-speculative-batch/internal/protocol"
	"opp300-speculative-batch/internal/store"
)

type submitJobRequest struct {
	IDempotencyKey string                `json:"idempotency_key"`
	Shards         []protocol.ShardInput `json:"shards"`
	Policy         protocol.Policy       `json:"policy"`
}

type submitJobResponse struct {
	Job           protocol.Job     `json:"job"`
	Shards        []protocol.Shard `json:"shards"`
	AlreadyExists bool             `json:"already_exists"`
}

func (s *Server) createJob(w http.ResponseWriter, r *http.Request) {
	var request submitJobRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "invalid JSON: "+err.Error())
		return
	}
	if request.IDempotencyKey == "" {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "idempotency_key is required")
		return
	}
	for i := range request.Shards {
		if request.Shards[i].Ordinal < 0 {
			writeError(w, http.StatusBadRequest, protocol.FailureValidation, "shard ordinal must be non-negative")
			return
		}
		if request.Shards[i].Behavior.FailureCategory == "UNKNOWN" {
			writeError(w, http.StatusBadRequest, protocol.FailureValidation, "UNKNOWN failure category is not accepted")
			return
		}
	}
	result, err := s.Store.CreateJob(r.Context(), protocol.JobSpec{
		IDempotencyKey: request.IDempotencyKey,
		Shards:         request.Shards,
		Policy:         request.Policy,
	})
	if err != nil {
		if errors.Is(err, store.ErrInvalidState) {
			writeError(w, http.StatusBadRequest, protocol.FailureValidation, err.Error())
			return
		}
		writeStoreError(w, err)
		return
	}
	s.Logger.Info("job submitted", "job_id", result.Job.ID, "input_id", request.IDempotencyKey,
		"total_shards", result.Job.TotalShards, "already_exists", result.AlreadyExists)
	writeJSON(w, http.StatusCreated, submitJobResponse{Job: result.Job, Shards: result.Shards, AlreadyExists: result.AlreadyExists})
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	job, err := s.Store.GetJob(r.Context(), r.PathValue("id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (s *Server) getProgress(w http.ResponseWriter, r *http.Request) {
	progress, err := s.Store.Progress(r.Context(), r.PathValue("id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, progress)
}

func (s *Server) listShards(w http.ResponseWriter, r *http.Request) {
	shards, err := s.Store.ListShards(r.Context(), r.PathValue("id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"shards": shards})
}

func (s *Server) listAttempts(w http.ResponseWriter, r *http.Request) {
	attempts, err := s.Store.ListAttempts(r.Context(), r.PathValue("id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"attempts": attempts, "coordinator_epoch": s.Epoch})
}

func (s *Server) listEvents(w http.ResponseWriter, r *http.Request) {
	after := int64(0)
	if raw := r.URL.Query().Get("after_id"); raw != "" {
		if parsed, ok := parsePositiveInt(raw); ok {
			after = int64(parsed)
		}
	}
	limit := 100
	if raw := r.URL.Query().Get("limit"); raw != "" {
		if parsed, ok := parsePositiveInt(raw); ok && parsed <= 1000 {
			limit = parsed
		}
	}
	events, err := s.Store.ListEvents(r.Context(), r.PathValue("id"), after, limit)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"events": events, "coordinator_epoch": s.Epoch})
}

func (s *Server) getResult(w http.ResponseWriter, r *http.Request) {
	result, err := s.Store.MergeResult(r.Context(), r.PathValue("id"), s.Artifacts)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
