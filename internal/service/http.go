package service

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"opp297/topk/internal/domain"
	"opp297/topk/internal/kernel"
)

type TopKRequest struct {
	Generation domain.GenerationID `json:"generation"`
	K          int                 `json:"k"`
}

type ShardStatusResponse struct {
	ID         int         `json:"id"`
	Exhausted  bool        `json:"exhausted"`
	Active     bool        `json:"active"`
	UpperBound *domain.Key `json:"upper_bound,omitempty"`
	Cursor     *domain.Key `json:"cursor,omitempty"`
	Pages      int         `json:"pages"`
	RowsRead   int         `json:"rows_read"`
}

type TopKResponse struct {
	RequestID  string                `json:"request_id"`
	Generation domain.GenerationID   `json:"generation"`
	K          int                   `json:"k"`
	Exact      bool                  `json:"exact"`
	Complete   bool                  `json:"complete"`
	Results    []domain.Candidate    `json:"results"`
	Threshold  *domain.Key           `json:"threshold,omitempty"`
	Failure    *FailureResponse      `json:"failure,omitempty"`
	Proof      StopProofResponse     `json:"stop_proof"`
	Shards     []ShardStatusResponse `json:"shard_statuses"`
}

type FailureResponse struct {
	Class   string `json:"class"`
	Message string `json:"message"`
}

type StopProofResponse struct {
	Rounds        int    `json:"rounds"`
	TotalRowsRead int    `json:"total_rows_read"`
	Reason        string `json:"reason"`
}

func (c *Coordinator) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/top-k", c.handleTopK)
	mux.HandleFunc("GET /v1/replay/{request_id}", c.handleReplay)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	return mux
}

func (c *Coordinator) handleTopK(w http.ResponseWriter, r *http.Request) {
	requestID := r.Header.Get("X-Request-ID")
	if requestID == "" {
		var err error
		requestID, err = NewRequestID()
		if err != nil {
			writeAPIError(w, http.StatusInternalServerError, err)
			return
		}
	}

	var req TopKRequest
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		writeAPIError(w, http.StatusBadRequest, err)
		return
	}
	if req.Generation == "" || req.K <= 0 || req.K > 10000 {
		writeAPIError(w, http.StatusBadRequest, errors.New("generation is required and k must be in [1,10000]"))
		return
	}

	result, err := c.Execute(r.Context(), requestID, req.Generation, req.K, r.Header.Get("X-Test-Fault"))
	if errors.Is(err, ErrUnknownGeneration) {
		writeAPIError(w, http.StatusNotFound, err)
		return
	}
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err)
		return
	}

	resp := TopKResponse{
		RequestID: requestID, Generation: req.Generation, K: req.K,
		Exact: result.Exact, Complete: result.Complete, Results: make([]domain.Candidate, 0, len(result.Keys)),
		Shards: make([]ShardStatusResponse, 0, len(result.Statuses)),
		Proof:  StopProofResponse{Rounds: result.Rounds, TotalRowsRead: result.TotalRows, Reason: finalReason(result.Events)},
	}
	for _, k := range result.Keys {
		resp.Results = append(resp.Results, domain.Candidate{ID: k.ID, Score: k.Score})
	}
	if len(result.Keys) == req.K {
		threshold := result.Keys[len(result.Keys)-1]
		resp.Threshold = &threshold
	}
	for _, st := range result.Statuses {
		resp.Shards = append(resp.Shards, ShardStatusResponse{
			ID: st.ID, Exhausted: st.Exhausted, Active: st.Active,
			UpperBound: st.UpperBound, Cursor: st.Cursor, Pages: st.Pages, RowsRead: st.RowsRead,
		})
	}
	if result.Failure != nil {
		resp.Failure = &FailureResponse{Class: string(result.Failure.Class), Message: result.Failure.Error()}
	}

	status := http.StatusOK
	if !result.Complete {
		status = http.StatusServiceUnavailable
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(resp)
}

func (c *Coordinator) handleReplay(w http.ResponseWriter, r *http.Request) {
	requestID := r.PathValue("request_id")
	replay, err := c.Replay(r.Context(), requestID)
	if errors.Is(err, ErrUnknownRequest) {
		writeAPIError(w, http.StatusNotFound, err)
		return
	}
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(replay)
}

func writeAPIError(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
}

func finalReason(events []kernel.Event) string {
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Kind == kernel.EventStop && events[i].Reason != "" {
			return events[i].Reason
		}
	}
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Kind == kernel.EventRejected {
			return events[i].Reason
		}
	}
	return ""
}
