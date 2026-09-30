package shardserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"topk-demo/internal/protocol"
	"topk-demo/internal/safolog"
	"topk-demo/internal/shardstore"
)

type Server struct {
	store shardstore.Store
	log   *slog.Logger
}

func New(store shardstore.Store, log *slog.Logger) *Server {
	return &Server{store: store, log: log}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/shards/bound", s.bound)
	mux.HandleFunc("POST /v1/shards/scan", s.scan)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	return requestIDMiddleware(s.log)(mux)
}

func (s *Server) bound(w http.ResponseWriter, r *http.Request) {
	var req protocol.BoundRequest
	if !decode(w, r, &req) {
		return
	}
	cursor, err := shardstore.DecodeCursor(req.Cursor)
	if err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureInvalidPayload, err.Error())
		return
	}
	resp, err := s.store.Bound(r.Context(), req.Generation, req.ShardID, req.Cursor, cursor)
	if err != nil {
		s.handleStoreError(w, r, err, req.ShardID, "bound")
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) scan(w http.ResponseWriter, r *http.Request) {
	var req protocol.ScanRequest
	if !decode(w, r, &req) {
		return
	}
	if req.PageSize <= 0 || req.PageSize > 1000 {
		writeError(w, http.StatusBadRequest, protocol.FailureInvalidPayload, "page_size must be between 1 and 1000")
		return
	}
	cursor, err := shardstore.DecodeCursor(req.Cursor)
	if err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureInvalidPayload, err.Error())
		return
	}
	resp, err := s.store.Scan(r.Context(), req.Generation, req.ShardID, req.Cursor, cursor, req.PageSize)
	if err != nil {
		s.handleStoreError(w, r, err, req.ShardID, "scan")
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleStoreError(w http.ResponseWriter, r *http.Request, err error, shardID, op string) {
	status := http.StatusBadGateway
	category := protocol.FailureNetwork
	if errors.Is(err, shardstore.ErrBadGeneration) {
		status = http.StatusConflict
		category = protocol.FailureBadGeneration
	}
	safolog.Warn(s.log, "shard call failed",
		"request_id", requestID(r.Context()),
		"op", op,
		"shard_id", shardID,
		"category", category,
		"error", err.Error())
	writeError(w, status, category, err.Error())
}

func decode(w http.ResponseWriter, r *http.Request, dst any) bool {
	defer r.Body.Close()
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeError(w, http.StatusBadRequest, protocol.FailureInvalidPayload, err.Error())
		return false
	}
	if dec.More() {
		writeError(w, http.StatusBadRequest, protocol.FailureInvalidPayload, "request body must contain exactly one JSON object")
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

type errorBody struct {
	Error    string `json:"error"`
	Category string `json:"category"`
}

func writeError(w http.ResponseWriter, status int, category, message string) {
	writeJSON(w, status, errorBody{Error: message, Category: category})
}

type requestIDKey struct{}

func requestIDMiddleware(log *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id := strings.TrimSpace(r.Header.Get("X-Request-ID"))
			if id == "" {
				id = fmt.Sprintf("local-%d", httpServerOffset.Add(1))
			}
			ctx := context.WithValue(r.Context(), requestIDKey{}, id)
			w.Header().Set("X-Request-ID", id)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func requestID(ctx context.Context) string {
	if id, ok := ctx.Value(requestIDKey{}).(string); ok {
		return id
	}
	return "unknown"
}
