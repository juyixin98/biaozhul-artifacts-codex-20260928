package coordinator

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"topk-demo/internal/protocol"
	"topk-demo/internal/safolog"
	"topk-demo/internal/state"
	"topk-demo/internal/topk"
)

type ShardConfig struct {
	ID  string `json:"id"`
	URL string `json:"url"`
}

type Config struct {
	Generation   string        `json:"generation"`
	DefaultK     int           `json:"default_k"`
	DefaultPage  int           `json:"default_page_size"`
	ShardTimeout time.Duration `json:"-"`
	Shards       []ShardConfig `json:"shards"`
}

type Service struct {
	cfg   Config
	store state.RunStore
	log   *slog.Logger
}

func New(cfg Config, store state.RunStore, log *slog.Logger) *Service {
	if cfg.DefaultK <= 0 {
		cfg.DefaultK = 10
	}
	if cfg.DefaultPage <= 0 {
		cfg.DefaultPage = 100
	}
	if cfg.ShardTimeout <= 0 {
		cfg.ShardTimeout = 750 * time.Millisecond
	}
	return &Service{cfg: cfg, store: store, log: log}
}

func (s *Service) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/topk", s.query)
	mux.HandleFunc("GET /v1/runs/{run_id}", s.replay)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	return requestIDMiddleware(s.log)(mux)
}

func (s *Service) query(w http.ResponseWriter, r *http.Request) {
	var req protocol.QueryRequest
	defer r.Body.Close()
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if req.K <= 0 || req.K > 10000 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "k must be between 1 and 10000"})
		return
	}
	generation := req.Generation
	if generation == "" {
		generation = s.cfg.Generation
	}
	if generation != s.cfg.Generation {
		writeJSON(w, http.StatusConflict, map[string]string{"error": "query generation is not active on this coordinator"})
		return
	}

	pageSize := req.PageSize
	if pageSize <= 0 {
		pageSize = s.cfg.DefaultPage
	}
	runID, _ := newRunID()
	ctx := context.WithValue(r.Context(), requestIDKey{}, requestIDFrom(r.Context(), runID))
	sink := &stateSink{store: s.store, runID: runID, generation: generation}
	client := &routedClient{shards: s.cfg.Shards}
	engine := topk.NewEngine(client)

	started := time.Now()
	_ = s.store.SaveRun(ctx, protocol.RunRecord{
		RunID:      runID,
		Generation: generation,
		K:          req.K,
		Complete:   false,
		Exact:      false,
		StopReason: "running",
		CreatedAt:  started,
	})
	result, err := engine.Execute(ctx, shardIDs(s.cfg.Shards), topk.Options{
		RunID:        runID,
		Generation:   generation,
		K:            req.K,
		PageSize:     pageSize,
		ShardTimeout: protocol.DurationFromMillis(req.ShardTimeoutMS, s.cfg.ShardTimeout),
		Sink:         sink,
	})
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	errSummary := ""
	if len(result.Errors) > 0 {
		parts := make([]string, 0, len(result.Errors))
		for _, item := range result.Errors {
			parts = append(parts, item.ShardID+":"+item.Category)
		}
		errSummary = strings.Join(parts, ";")
	}
	_ = s.store.SaveRun(ctx, protocol.RunRecord{
		RunID:                runID,
		Generation:           generation,
		K:                    req.K,
		Complete:             result.Complete,
		Exact:                result.Exact,
		ResultCount:          len(result.Results),
		StopReason:           result.StopReason,
		BoundProbes:          result.Metrics.BoundProbes,
		StorageRowsRead:      result.Metrics.StorageRowsRead,
		CandidatesReturned:   result.Metrics.CandidatesReturned,
		CandidatesConsidered: result.Metrics.CandidatesConsidered,
		ErrorSummary:         errSummary,
		CreatedAt:            started,
	})
	safolog.Info(s.log, "query complete",
		"request_id", requestIDFrom(ctx, runID),
		"run_id", runID,
		"generation", generation,
		"exact", result.Exact,
		"complete", result.Complete,
		"stop_reason", result.StopReason,
		"storage_rows_read", result.Metrics.StorageRowsRead,
		"candidates_returned", result.Metrics.CandidatesReturned)
	writeJSON(w, http.StatusOK, result)
}

func (s *Service) replay(w http.ResponseWriter, r *http.Request) {
	runID := r.PathValue("run_id")
	run, events, err := s.store.GetRun(r.Context(), runID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"run": run, "events": events})
}

type routedClient struct{ shards []ShardConfig }

func (c *routedClient) Bound(ctx context.Context, req protocol.BoundRequest) (*protocol.BoundResponse, error) {
	url, err := c.url(req.ShardID)
	if err != nil {
		return nil, &topk.ShardCallError{Category: protocol.FailureInvalidShard, Message: err.Error()}
	}
	return NewHTTPShardClient(url).Bound(ctx, req)
}

func (c *routedClient) Scan(ctx context.Context, req protocol.ScanRequest) (*protocol.ScanResponse, error) {
	url, err := c.url(req.ShardID)
	if err != nil {
		return nil, &topk.ShardCallError{Category: protocol.FailureInvalidShard, Message: err.Error()}
	}
	return NewHTTPShardClient(url).Scan(ctx, req)
}

func (c *routedClient) url(id string) (string, error) {
	for _, shard := range c.shards {
		if shard.ID == id {
			return shard.URL, nil
		}
	}
	return "", fmt.Errorf("shard %q is not configured", id)
}

func shardIDs(shards []ShardConfig) []string {
	ids := make([]string, 0, len(shards))
	for _, shard := range shards {
		ids = append(ids, shard.ID)
	}
	return ids
}

type stateSink struct {
	store      state.RunStore
	runID      string
	generation string
}

func (s *stateSink) Record(ctx context.Context, ev protocol.EventRecord) error {
	if ev.CreatedAt.IsZero() {
		ev.CreatedAt = time.Now()
	}
	return s.store.SaveEvent(ctx, ev)
}

func newRunID() (string, error) {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return "run_" + hex.EncodeToString(b[:]), nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
