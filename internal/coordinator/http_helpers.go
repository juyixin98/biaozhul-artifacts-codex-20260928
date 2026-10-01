package coordinator

import (
	"encoding/json"
	"errors"
	"net/http"

	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/store"
)

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func epochHeader(epoch string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Coordinator-Epoch", epoch)
		next.ServeHTTP(w, r)
	})
}

func requestLog(logger interface {
	Info(string, ...any)
}, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recorder := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(recorder, r)
		logger.Info("http_request",
			"method", r.Method, "path", r.URL.Path, "status", recorder.status,
			"job_id", r.PathValue("id"), "attempt_id", r.PathValue("attemptID"),
			"remote", r.RemoteAddr)
	})
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

func writeStoreError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, store.ErrNotFound):
		writeError(w, http.StatusNotFound, "NOT_FOUND", err.Error())
	case errors.Is(err, store.ErrFenced):
		writeError(w, http.StatusConflict, "INTERRUPTED", err.Error())
	case errors.Is(err, store.ErrSuperseded):
		writeError(w, http.StatusConflict, "SUPERSEDED", err.Error())
	case errors.Is(err, store.ErrInvalidState), errors.Is(err, store.ErrChecksum):
		writeError(w, http.StatusConflict, "INVALID_STATE", err.Error())
	default:
		writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
	}
}
