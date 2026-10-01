package worker

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"

	"opp300-speculative-batch/internal/participant"
)

type Server struct {
	Kernel *Kernel
	Logger *slog.Logger

	baseCtx context.Context
	mu      sync.Mutex
	active  map[string]context.CancelFunc
}

func NewServer(kernel *Kernel, logger *slog.Logger) *Server {
	return &Server{
		Kernel:  kernel,
		Logger:  logger,
		baseCtx: context.Background(),
		active:  map[string]context.CancelFunc{},
	}
}

func (s *Server) WithBaseContext(ctx context.Context) *Server {
	s.baseCtx = ctx
	return s
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /execute", s.execute)
	return mux
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) execute(w http.ResponseWriter, r *http.Request) {
	var req participant.ExecuteRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "VALIDATION", err.Error())
		return
	}
	s.mu.Lock()
	if _, exists := s.active[req.AttemptID]; exists {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, "VALIDATION", "attempt is already executing on this worker")
		return
	}
	ctx, cancel := context.WithCancel(WithServerURL(s.baseCtx, serverBaseURL(r)))
	s.active[req.AttemptID] = cancel
	s.mu.Unlock()

	s.Logger.Info("attempt accepted",
		"job_id", req.JobID, "shard_key", req.Key, "attempt_id", req.AttemptID,
		"attempt_number", req.AttemptNumber, "kind", req.Kind, "epoch", req.Epoch,
		"delay_ms", req.Behavior.DelayMS, "should_fail", req.Behavior.Fail)

	go func() {
		defer func() {
			s.mu.Lock()
			delete(s.active, req.AttemptID)
			s.mu.Unlock()
			cancel()
		}()
		if err := s.Kernel.Run(ctx, req); err != nil {
			s.Logger.Error("attempt finished with error", "attempt_id", req.AttemptID, "error", err)
		} else {
			s.Logger.Info("attempt processed", "attempt_id", req.AttemptID, "job_id", req.JobID, "shard_key", req.Key)
		}
	}()

	writeJSON(w, http.StatusAccepted, map[string]string{"status": "accepted", "attempt_id": req.AttemptID})
}

func (s *Server) Shutdown() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, cancel := range s.active {
		cancel()
	}
	s.active = map[string]context.CancelFunc{}
}

func serverBaseURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return scheme + "://" + r.Host
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	return decoder.Decode(dst)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, category, message string) {
	writeJSON(w, status, participant.ErrorResponse{Error: message, FailureCategory: category})
}
