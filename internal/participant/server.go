package participant

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"time"

	"opp297/topk/internal/domain"
	"opp297/topk/internal/kernel"
	"opp297/topk/internal/storage"
)

type Server struct {
	Store       *storage.Store
	ShardID     int
	BoundSecret []byte
	Fault       string
	AllowFault  bool
	Logger      *log.Logger
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/shards/page", s.readPage)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	return mux
}

func (s *Server) readPage(w http.ResponseWriter, r *http.Request) {
	requestID := r.Header.Get("X-Request-ID")
	var req domain.PageRequest
	if err := decodeJSON(r, &req); err != nil {
		s.writeError(w, requestID, http.StatusBadRequest, err)
		return
	}
	if req.ShardID != s.ShardID {
		err := errors.New("request is bound to a different local shard")
		s.writeError(w, requestID, http.StatusBadRequest, err)
		return
	}
	if req.Limit <= 0 || req.Limit > 1000 {
		s.writeError(w, requestID, http.StatusBadRequest, errors.New("limit must be in [1,1000]"))
		return
	}

	fault := s.Fault
	if s.AllowFault && r.Header.Get("X-Test-Fault") != "" {
		fault = r.Header.Get("X-Test-Fault")
	}
	switch fault {
	case "slow":
		select {
		case <-time.After(5 * time.Second):
		case <-r.Context().Done():
			s.logReject(requestID, req, "slow fault exceeded caller deadline")
			s.writeError(w, requestID, http.StatusGatewayTimeout, r.Context().Err())
			return
		}
	case "wrong_generation":
	}

	rows, err := s.Store.ReadItemPage(r.Context(), req.Generation, req.ShardID, req.Limit, req.Cursor)
	if err != nil {
		s.logReject(requestID, req, err.Error())
		s.writeError(w, requestID, http.StatusBadGateway, err)
		return
	}

	page := domain.PageResponse{
		Generation: req.Generation,
		ShardID:    s.ShardID,
		RowsRead:   len(rows),
		Items:      make([]domain.Candidate, 0, req.Limit),
	}
	for _, row := range rows {
		if len(page.Items) == req.Limit {
			break
		}
		page.Items = append(page.Items, row.Item)
	}
	if len(page.Items) > 0 {
		last := page.Items[len(page.Items)-1].Key()
		page.Cursor = &last
	}
	if len(rows) > req.Limit {
		page.HasMore = true
		bound := rows[req.Limit].Item.Key()
		page.UpperBound = &bound
		page.BoundProof = kernel.SignBound(s.BoundSecret,
			kernel.BoundPayload(req.Generation, s.ShardID, *page.Cursor, bound))
	}

	switch fault {
	case "wrong_generation":
		page.Generation = domain.GenerationID(string(req.Generation) + "-wrong")
	case "bad_bound":
		if page.UpperBound != nil {
			page.UpperBound.Score--
		}
	case "forged_bound":
		if page.UpperBound != nil {
			page.BoundProof = "forged"
		}
	case "bad_order":
		if len(page.Items) > 1 {
			page.Items[0], page.Items[1] = page.Items[1], page.Items[0]
		}
	}

	s.logAccept(requestID, req, page)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(page); err != nil {
		s.Logger.Printf("request_id=%s shard=%d encode_error=%v", requestID, s.ShardID, err)
	}
}

func (s *Server) writeError(w http.ResponseWriter, requestID string, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"request_id": requestID,
		"error":      err.Error(),
	})
}

func (s *Server) logAccept(requestID string, req domain.PageRequest, page domain.PageResponse) {
	s.Logger.Printf("request_id=%s shard=%d gen=%s accepted items=%d rows=%d has_more=%t bound=%s",
		requestID, s.ShardID, redactGeneration(req.Generation), len(page.Items), page.RowsRead, page.HasMore, keyLabel(page.UpperBound))
}

func (s *Server) logReject(requestID string, req domain.PageRequest, reason string) {
	s.Logger.Printf("request_id=%s shard=%d gen=%s rejected reason=%q",
		requestID, s.ShardID, redactGeneration(req.Generation), reason)
}

func decodeJSON(r *http.Request, v any) error {
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(v); err != nil {
		return err
	}
	return nil
}

func redactGeneration(gen domain.GenerationID) string {
	if gen == "" {
		return "-"
	}
	text := string(gen)
	if len(text) <= 8 {
		return text[:1] + "***"
	}
	return text[:4] + "***" + text[len(text)-4:]
}

func keyLabel(k *domain.Key) string {
	if k == nil {
		return "-"
	}
	return k.String()
}
