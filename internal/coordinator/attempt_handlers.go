package coordinator

import (
	"errors"
	"io"
	"net/http"

	"opp300-speculative-batch/internal/artifacts"
	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/protocol"
	"opp300-speculative-batch/internal/store"
)

func (s *Server) startAttempt(w http.ResponseWriter, r *http.Request) {
	var request participant.StartRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "invalid JSON: "+err.Error())
		return
	}
	attempt, err := s.Store.GetAttempt(r.Context(), r.PathValue("id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	policy, err := s.Store.GetPolicy(r.Context(), attempt.JobID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	updated, err := s.Store.MarkAttemptRunning(r.Context(), attempt.ID, store.RunningInput{
		Now:       s.Now(),
		Epoch:     request.Epoch,
		LeaseFor:  durationMS(policy.LeaseMS),
		WorkerURL: request.WorkerURL,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, updated)
}

func (s *Server) heartbeat(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Epoch string `json:"epoch"`
	}
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "invalid JSON: "+err.Error())
		return
	}
	existing, err := s.Store.GetAttempt(r.Context(), r.PathValue("id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	policy, err := s.Store.GetPolicy(r.Context(), existing.JobID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	attempt, err := s.Store.Heartbeat(r.Context(), existing.ID, request.Epoch, s.Now(), durationMS(policy.LeaseMS))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, attempt)
}

func (s *Server) failAttempt(w http.ResponseWriter, r *http.Request) {
	var request participant.FailureRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "invalid JSON: "+err.Error())
		return
	}
	if request.FailureCategory == "" || request.FailureCategory == protocol.FailureSuperseded {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "failure_category must be a concrete non-superseded failure")
		return
	}
	attempt, err := s.Store.FailAttempt(r.Context(), r.PathValue("id"), store.AttemptFailure{
		Now:       s.Now(),
		Epoch:     request.Epoch,
		Category:  request.FailureCategory,
		Message:   request.FailureMessage,
		WorkerURL: request.WorkerURL,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, attempt)
}

func (s *Server) stageArtifact(w http.ResponseWriter, r *http.Request) {
	checksum := r.URL.Query().Get("checksum_sha256")
	epoch := r.URL.Query().Get("epoch")
	if checksum == "" {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "checksum_sha256 query parameter is required")
		return
	}
	attempt, err := s.Store.GetAttempt(r.Context(), r.PathValue("id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if attempt.Epoch != epoch {
		writeError(w, http.StatusConflict, protocol.FailureInterrupted, "artifact upload belongs to a fenced coordinator epoch")
		return
	}
	staged, err := s.Artifacts.Stage(io.LimitReader(r.Body, s.MaxArtifact+1), checksum, s.MaxArtifact)
	if err != nil {
		if errors.Is(err, artifacts.ErrChecksumMismatch) || errors.Is(err, artifacts.ErrInvalidChecksum) {
			writeError(w, http.StatusBadRequest, protocol.FailureValidation, err.Error())
			return
		}
		writeStoreError(w, err)
		return
	}
	s.Staging.Put(staged)
	s.Logger.Info("artifact staged", "attempt_id", r.PathValue("id"),
		"staging_id", staged.ID, "checksum_sha256", staged.ChecksumSHA256, "size_bytes", staged.SizeBytes)
	writeJSON(w, http.StatusCreated, participant.StageResponse{
		StagingID:      staged.ID,
		ChecksumSHA256: staged.ChecksumSHA256,
		SizeBytes:      staged.SizeBytes,
	})
}

func (s *Server) completeAttempt(w http.ResponseWriter, r *http.Request) {
	var request participant.CompleteRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "invalid JSON: "+err.Error())
		return
	}
	staged, ok := s.Staging.Get(request.StagingID)
	if !ok || staged.ChecksumSHA256 != request.ChecksumSHA256 || staged.SizeBytes != request.SizeBytes {
		writeError(w, http.StatusBadRequest, protocol.FailureValidation, "staging id, checksum, or size does not match this coordinator process")
		return
	}
	result, err := s.Store.CompleteAttempt(r.Context(), store.CompleteInput{
		AttemptID: r.PathValue("id"),
		Epoch:     request.Epoch,
		Now:       s.Now(),
		Staged:    staged,
		Publisher: s.Artifacts,
	})
	if err != nil {
		if errors.Is(err, store.ErrSuperseded) || errors.Is(err, store.ErrFenced) ||
			errors.Is(err, store.ErrInvalidState) || errors.Is(err, store.ErrChecksum) {
			s.Staging.Delete(request.StagingID)
			s.Artifacts.RemoveStaged(staged)
		}
		writeStoreError(w, err)
		return
	}
	s.Staging.Delete(request.StagingID)
	s.Artifacts.RemoveStaged(staged)
	s.Logger.Info("artifact published", "job_id", result.Job.ID, "shard_id", result.Shard.ID,
		"attempt_id", result.Attempt.ID, "winner_kind", result.Attempt.Kind,
		"job_version", result.Job.Version, "job_state", result.Job.State,
		"checksum_sha256", request.ChecksumSHA256)
	writeJSON(w, http.StatusOK, result)
}
