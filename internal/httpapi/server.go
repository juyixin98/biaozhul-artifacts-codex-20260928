package httpapi

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"diststats/internal/protocol"
	"diststats/internal/service"
)

const maxBodyBytes = 4 << 20

type Server struct {
	service *service.Service
	logger  *slog.Logger
	mux     *http.ServeMux
}

func New(svc *service.Service, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	server := &Server{service: svc, logger: logger, mux: http.NewServeMux()}
	server.routes()
	return server
}

func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) routes() {
	s.mux.HandleFunc("POST /v1/jobs", s.createJob)
	s.mux.HandleFunc("GET /v1/jobs/{id}", s.getJob)
	s.mux.HandleFunc("POST /v1/jobs/{id}/shards", s.submitShard)
	s.mux.HandleFunc("GET /v1/jobs/{id}/replay", s.replay)
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
}

func (s *Server) createJob(w http.ResponseWriter, r *http.Request) {
	var request protocol.JobRequest
	if err := decodeJSON(w, r, &request); err != nil {
		s.writeError(w, r, err, "", "", "")
		return
	}
	result, err := s.service.CreateJob(r.Context(), request)
	if err != nil {
		s.writeError(w, r, err, request.RunID, "", "")
		return
	}
	s.logger.LogAttrs(r.Context(), slog.LevelInfo, "job_created",
		slog.String("run_id", result.Job.RunID),
		slog.String("job_id", result.Job.ID),
		slog.Int("dimension", result.Job.Dimension),
		slog.Int("expected_shards", result.Job.ExpectedShards))
	writeJSON(w, http.StatusCreated, result)
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	result, err := s.service.GetJob(r.Context(), jobID)
	if err != nil {
		s.writeError(w, r, err, "", jobID, "")
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) submitShard(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	var request protocol.ShardRequest
	if err := decodeJSON(w, r, &request); err != nil {
		s.writeError(w, r, err, "", jobID, request.ShardID)
		return
	}
	result, err := s.service.SubmitShard(r.Context(), jobID, request)
	if err != nil {
		s.writeError(w, r, err, "", jobID, request.ShardID)
		return
	}
	s.logger.LogAttrs(r.Context(), slog.LevelInfo, "shard_accepted",
		slog.String("job_id", jobID),
		slog.String("shard_id", request.ShardID),
		slog.Int64("sequence", result.Sequence),
		slog.Int("observations", result.Observation),
		slog.Float64("weight", result.Weight),
		slog.String("decision", "unique shard merged"))
	writeJSON(w, http.StatusAccepted, result)
}

func (s *Server) replay(w http.ResponseWriter, r *http.Request) {
	events, result, err := s.service.Replay(r.Context(), r.PathValue("id"))
	if err != nil {
		s.writeError(w, r, err, "", r.PathValue("id"), "")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"result": result,
		"events": events,
		"replay": map[string]any{
			"event_count": len(events),
			"basis":       "events are returned in sequence and include before/after state and the decision reason",
		},
	})
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			return protocol.NewServiceError(protocol.ClassResourceExhausted, "REQUEST_TOO_LARGE",
				"request body exceeds the configured 4 MiB local limit", err)
		}
		return protocol.NewServiceError(protocol.ClassInput, "INVALID_JSON",
			"request body must be valid JSON matching the documented schema", err)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return protocol.NewServiceError(protocol.ClassInput, "INVALID_JSON",
			"request body must contain exactly one JSON value", nil)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func (s *Server) writeError(w http.ResponseWriter, r *http.Request, err error, runID, jobID, shardID string) {
	serviceErr, ok := protocol.AsServiceError(err)
	if !ok {
		serviceErr = protocol.NewServiceError(protocol.ClassComputation, "COMPUTATION_FAILED", err.Error(), err)
	}
	if runID == "" {
		runID = r.URL.Query().Get("run_id")
	}
	status := statusForClass(serviceErr.Class)
	if serviceErr.Code == "REQUEST_TOO_LARGE" {
		status = http.StatusRequestEntityTooLarge
	}
	response := protocol.ErrorResponse{
		ErrorClass: string(serviceErr.Class),
		Code:       serviceErr.Code,
		Message:    serviceErr.Message,
		RunID:      runID,
		JobID:      jobID,
		ShardID:    shardID,
	}
	s.logger.LogAttrs(r.Context(), slog.LevelWarn, "request_failed",
		slog.String("run_id", runID),
		slog.String("job_id", jobID),
		slog.String("shard_id", shardID),
		slog.String("error_class", response.ErrorClass),
		slog.String("code", response.Code),
		slog.Int("http_status", status),
		slog.String("path", r.URL.Path),
		slog.String("decision", strings.ReplaceAll(response.ErrorClass+"_"+response.Code, "_", " ")))
	writeJSON(w, status, response)
}

func statusForClass(class protocol.ErrorClass) int {
	switch class {
	case protocol.ClassInput:
		return http.StatusBadRequest
	case protocol.ClassState:
		return http.StatusConflict
	case protocol.ClassNotFound:
		return http.StatusNotFound
	case protocol.ClassResourceExhausted:
		return http.StatusInsufficientStorage
	case protocol.ClassComputation:
		return http.StatusUnprocessableEntity
	default:
		return http.StatusInternalServerError
	}
}
