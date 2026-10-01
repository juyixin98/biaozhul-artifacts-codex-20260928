package httpapi

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"diststats/internal/apperror"
	"diststats/internal/service"
)

type Server struct {
	service *service.Service
	log     *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) http.Handler {
	s := &Server{service: svc, log: log}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /v1/jobs", s.createJob)
	mux.HandleFunc("GET /v1/jobs/{id}", s.getJob)
	mux.HandleFunc("POST /v1/jobs/{id}/shards", s.submitShard)
	mux.HandleFunc("POST /v1/jobs/{id}/finalize", s.finalizeJob)
	mux.HandleFunc("GET /v1/jobs/{id}/result", s.getResult)
	mux.HandleFunc("GET /v1/jobs/{id}/replay", s.replay)
	mux.HandleFunc("GET /v1/jobs/{id}/runs", s.listJobRuns)
	mux.HandleFunc("GET /v1/runs/{number}", s.getRun)
	return mux
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, err error) {
	ae := &apperror.Error{}
	if !errors.As(err, &ae) {
		ae = apperror.Unavailable("INTERNAL", "unexpected internal error", err)
	}
	writeJSON(w, statusForKind(ae.Kind), map[string]any{"error": struct {
		Kind    string `json:"kind"`
		Code    string `json:"code"`
		Message string `json:"message"`
	}{Kind: string(ae.Kind), Code: ae.Code, Message: ae.Message}})
}

func statusForKind(kind apperror.Kind) int {
	switch kind {
	case apperror.KindInvalidInput:
		return http.StatusBadRequest
	case apperror.KindStateConflict:
		return http.StatusConflict
	case apperror.KindNotFound:
		return http.StatusNotFound
	case apperror.KindResourceExhausted:
		return http.StatusUnprocessableEntity
	case apperror.KindComputationFailed:
		return http.StatusUnprocessableEntity
	default:
		return http.StatusServiceUnavailable
	}
}
