package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	"opp299/internal/kernel"
)

type requestIDKey struct{}

type errorBody struct {
	RequestID string `json:"request_id"`
	Category  string `json:"category"`
	Message   string `json:"message"`
	Uncertain bool   `json:"uncertain"`
}

func (s *Server) requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			var raw [8]byte
			_, _ = rand.Read(raw[:])
			id = "req-" + hex.EncodeToString(raw[:])
		}
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *Server) recoverLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		defer func() {
			if recovered := recover(); recovered != nil {
				log.Printf("request_id=%s method=%s path=%s status=500 panic=%v", requestIDFromContext(r.Context()), r.Method, r.URL.Path, recovered)
				writeJSON(w, http.StatusInternalServerError, errorBody{
					RequestID: requestIDFromContext(r.Context()),
					Category:  string(kernel.CategoryStorage),
					Message:   fmt.Sprintf("internal panic: %v", recovered),
					Uncertain: true,
				})
				return
			}
			log.Printf("request_id=%s method=%s path=%s status=%d duration=%s", requestIDFromContext(r.Context()), r.Method, r.URL.Path, sw.status, time.Since(start))
		}()
		next.ServeHTTP(sw, r)
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func requestIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey{}).(string)
	return id
}

func decode(r *http.Request, dst any) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if dec.More() {
		return errors.New("request body contains multiple JSON documents")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, r *http.Request, status int, category kernel.ErrorCategory, err error) {
	body := errorBody{
		RequestID: requestIDFromContext(r.Context()),
		Category:  string(category),
		Message:   err.Error(),
		Uncertain: category == kernel.CategoryStorage,
	}
	writeJSON(w, status, body)
}

func writeStorageError(w http.ResponseWriter, r *http.Request, err error) {
	classified := kernel.Classify(err)
	status := http.StatusInternalServerError
	if classified.Category == kernel.CategoryNotFound {
		status = http.StatusNotFound
	}
	if classified.Category == kernel.CategoryProtocol {
		status = http.StatusConflict
	}
	writeError(w, r, status, classified.Category, classified)
}

func parseID(w http.ResponseWriter, r *http.Request) (int64, bool) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		writeError(w, r, http.StatusBadRequest, kernel.CategoryProtocol, fmt.Errorf("checkpoint id must be a positive integer"))
		return 0, false
	}
	return id, true
}
