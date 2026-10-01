package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"specbatch/internal/kernel"
	"specbatch/internal/logging"
	"specbatch/internal/model"
	"specbatch/internal/store"
)

type Server struct {
	store   *store.Store
	dir     string
	epoch   int64
	log     *logging.Logger
	version string
}

func NewServer(st *store.Store, artifactDir string, epoch int64, log *logging.Logger) *Server {
	return &Server{store: st, dir: artifactDir, epoch: epoch, log: log, version: "coordinator/v1.0.0"}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /v1/jobs", s.createJob)
	mux.HandleFunc("GET /v1/jobs/{jobID}", s.getJob)
	mux.HandleFunc("GET /v1/jobs/{jobID}/merge", s.merge)
	mux.HandleFunc("GET /v1/events", s.events)
	mux.HandleFunc("POST /internal/v1/poll", s.poll)
	mux.HandleFunc("POST /internal/v1/heartbeat", s.heartbeat)
	mux.HandleFunc("POST /internal/v1/fail", s.fail)
	mux.HandleFunc("POST /internal/v1/complete", s.complete)
	return mux
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok", "version": s.version, "protocol": "specbatch/v1",
		"kernel": kernel.Name + "@" + kernel.Version, "epoch": s.epoch,
	})
}

type createJobRequest struct {
	Manifest model.Manifest `json:"manifest"`
}

func (s *Server) createJob(w http.ResponseWriter, r *http.Request) {
	var req createJobRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, model.FailureBadRequest, err.Error())
		return
	}
	if err := s.store.CreateJob(r.Context(), req.Manifest); err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, store.ErrConflict) {
			status = http.StatusConflict
		}
		if errors.Is(err, store.ErrValidation) {
			status = http.StatusBadRequest
		}
		writeError(w, status, model.FailureValidation, err.Error())
		return
	}
	s.log.Info("job_submitted", map[string]any{
		"job_id": req.Manifest.JobID, "shards": len(req.Manifest.Shards),
		"spec_budget": req.Manifest.MaxSpeculativeCopies, "run_epoch": s.epoch,
	})
	writeJSON(w, http.StatusCreated, map[string]string{"job_id": req.Manifest.JobID, "status": model.JobQueued})
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	includeAttempts := r.URL.Query().Get("attempts") == "1"
	st, err := s.store.JobStatus(r.Context(), r.PathValue("jobID"), includeAttempts)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "job_not_found", "job does not exist")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) merge(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("jobID")
	if r.URL.Query().Get("persist") == "1" {
		if err := s.PublishArtifacts(r.Context(), jobID); err != nil {
			status := http.StatusBadRequest
			if !errors.Is(err, errJobNotComplete) {
				status = http.StatusInternalServerError
			}
			writeError(w, status, model.FailureUnknown, err.Error())
			return
		}
	}
	doc, err := s.store.Merged(r.Context(), jobID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "merge_unavailable", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

func (s *Server) events(w http.ResponseWriter, r *http.Request) {
	jobID := r.URL.Query().Get("job_id")
	events, err := s.store.Events(r.Context(), jobID, 500)
	if err != nil {
		writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"events": events})
}

type workerRequest struct {
	WorkerID string `json:"worker_id"`
}

type heartbeatRequest struct {
	Token    string  `json:"token"`
	WorkerID string  `json:"worker_id"`
	Progress float64 `json:"progress"`
}

type failRequest struct {
	Token    string `json:"token"`
	WorkerID string `json:"worker_id"`
	Category string `json:"category"`
	Message  string `json:"message"`
}

func (s *Server) poll(w http.ResponseWriter, r *http.Request) {
	var req workerRequest
	if err := decodeJSON(r, &req); err != nil || req.WorkerID == "" {
		writeError(w, http.StatusBadRequest, model.FailureBadRequest, "worker_id required")
		return
	}
	deadline := time.Now().Add(900 * time.Millisecond)
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		claim, ok, err := s.store.Claim(r.Context(), req.WorkerID, s.epoch)
		if err != nil {
			writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
			return
		}
		if ok {
			writeJSON(w, http.StatusOK, map[string]any{"lease": claim.Lease, "epoch": claim.Epoch})
			return
		}
		if time.Now().After(deadline) {
			writeJSON(w, http.StatusNoContent, nil)
			return
		}
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
		}
	}
}

func (s *Server) heartbeat(w http.ResponseWriter, r *http.Request) {
	var req heartbeatRequest
	if err := decodeJSON(r, &req); err != nil || req.Token == "" || req.WorkerID == "" {
		writeError(w, http.StatusBadRequest, model.FailureBadRequest, "token and worker_id required")
		return
	}
	epoch, err := s.store.Heartbeat(r.Context(), req.Token, req.WorkerID, req.Progress)
	if err != nil {
		s.writeAttemptError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "epoch": epoch, "progress": req.Progress})
}

func (s *Server) fail(w http.ResponseWriter, r *http.Request) {
	var req failRequest
	if err := decodeJSON(r, &req); err != nil || req.Token == "" || req.WorkerID == "" {
		writeError(w, http.StatusBadRequest, model.FailureBadRequest, "token, worker_id and failure category required")
		return
	}
	if err := s.store.ReportFailure(r.Context(), req.Token, req.WorkerID, req.Category, req.Message); err != nil {
		s.writeAttemptError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"accepted": true, "failure_category": req.Category})
}

func (s *Server) complete(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		writeError(w, http.StatusBadRequest, model.FailureBadRequest, err.Error())
		return
	}
	token := strings.TrimSpace(r.FormValue("token"))
	workerID := strings.TrimSpace(r.FormValue("worker_id"))
	if token == "" || workerID == "" {
		writeError(w, http.StatusBadRequest, model.FailureBadRequest, "token and worker_id required")
		return
	}
	file, _, err := r.FormFile("artifact")
	if err != nil {
		writeError(w, http.StatusBadRequest, model.FailureBadRequest, "artifact file required")
		return
	}
	defer file.Close()
	tmp, err := os.CreateTemp(s.dir, ".upload-*")
	if err != nil {
		_ = os.MkdirAll(s.dir, 0o755)
		tmp, err = os.CreateTemp(s.dir, ".upload-*")
		if err != nil {
			writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
			return
		}
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	defer tmp.Close()
	if _, err := io.Copy(tmp, file); err != nil {
		writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
		return
	}
	if err := tmp.Close(); err != nil {
		writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
		return
	}
	raw, err := os.ReadFile(tmpName)
	if err != nil {
		writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
		return
	}
	var doc model.ArtifactDocument
	if err := json.Unmarshal(raw, &doc); err != nil {
		writeError(w, http.StatusBadRequest, model.FailureValidation, err.Error())
		return
	}
	result, err := s.store.SubmitSuccess(r.Context(), store.SubmitInput{
		Token: token, WorkerID: workerID, Document: doc, Content: raw,
	})
	if err != nil {
		s.writeAttemptError(w, err)
		return
	}
	if result.Won {
		_ = s.PublishOne(r.Context(), result.JobID, result.ShardID)
		_ = s.PublishMerge(r.Context(), result.JobID)
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) writeAttemptError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, store.ErrInvalidToken):
		writeError(w, http.StatusUnauthorized, model.FailureInvalidToken, err.Error())
	case errors.Is(err, store.ErrAttemptExpired):
		writeError(w, http.StatusConflict, model.FailureSupersededSubmission, err.Error())
	case errors.Is(err, store.ErrValidation):
		writeError(w, http.StatusUnprocessableEntity, model.FailureValidation, err.Error())
	default:
		writeError(w, http.StatusInternalServerError, model.FailureUnknown, err.Error())
	}
}

func decodeJSON(r *http.Request, v any) error {
	dec := json.NewDecoder(io.LimitReader(r.Body, 4<<20))
	dec.DisallowUnknownFields()
	return dec.Decode(v)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	if status == http.StatusNoContent || v == nil {
		w.WriteHeader(status)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

type ErrorBody struct {
	Error           string `json:"error"`
	FailureCategory string `json:"failure_category"`
}

func writeError(w http.ResponseWriter, status int, category, message string) {
	writeJSON(w, status, ErrorBody{Error: message, FailureCategory: category})
}

func writeFile(path string, content []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	final, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer final.Close()
	if _, err := final.Write(content); err != nil {
		return err
	}
	return nil
}

var _ = context.Background
