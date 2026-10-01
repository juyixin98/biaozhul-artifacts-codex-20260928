package service

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"

	"diststats/internal/apperror"
	"diststats/internal/compute"
	"diststats/internal/id"
	"diststats/internal/protocol"
	"diststats/internal/store"
)

type Limits struct {
	MaxDimensions   int
	MaxObservations int
}

type Service struct {
	db     *store.DB
	limits Limits
}

func New(db *store.DB, limits Limits) *Service {
	return &Service{db: db, limits: limits}
}

type RunHandle struct {
	ID        string
	Number    int64
	route     string
	method    string
	eventNext int64
}

type EventState struct {
	Before    protocol.SummaryDTO  `json:"before,omitempty"`
	After     protocol.SummaryDTO  `json:"after,omitempty"`
	Shard     *protocol.SummaryDTO `json:"shard,omitempty"`
	ShardID   string               `json:"shard_id,omitempty"`
	Seq       int64                `json:"seq,omitempty"`
	Reason    string               `json:"reason,omitempty"`
	Dimension int                  `json:"dimension,omitempty"`
}

func (s *Service) StartRun(ctx context.Context, method, route string, request any, jobID *string) (*RunHandle, error) {
	runID, err := id.New()
	if err != nil {
		return nil, err
	}
	var req json.RawMessage
	if request != nil {
		req = protocol.RawJSON(request)
	}
	number, err := s.db.InsertRun(ctx, store.Run{
		ID:        runID,
		JobID:     jobID,
		Route:     route,
		Method:    method,
		Status:    "started",
		Request:   req,
		StartedAt: time.Now().UTC(),
	})
	if err != nil {
		return nil, err
	}
	return &RunHandle{ID: runID, Number: number, route: route, method: method, eventNext: 1}, nil
}

func (s *Service) CompleteRun(ctx context.Context, h *RunHandle, statusCode int, err error) error {
	status := "succeeded"
	var payload json.RawMessage
	if err != nil {
		status = "failed"
		payload = errorJSON(err)
	}
	return s.db.CompleteRun(ctx, h.ID, statusCode, status, payload, time.Now().UTC())
}

func (s *Service) AddEvent(ctx context.Context, h *RunHandle, stage, result, reason string, state any) error {
	b := protocol.RawJSON(state)
	if err := s.db.AddRunEvent(ctx, store.RunEvent{
		RunID:     h.ID,
		Index:     h.eventNext,
		Stage:     stage,
		Result:    result,
		Reason:    reason,
		State:     b,
		CreatedAt: time.Now().UTC(),
	}); err != nil {
		return err
	}
	h.eventNext++
	return nil
}

func addTxEvent(ctx context.Context, tx *sql.Tx, h *RunHandle, stage, result, reason string, state any) error {
	b := protocol.RawJSON(state)
	_, err := tx.ExecContext(ctx, `
INSERT INTO run_events (run_id, event_index, stage, result, reason, state)
VALUES ($1, $2, $3, $4, $5, $6)`,
		h.ID, h.eventNext, stage, result, reason, []byte(b))
	if err != nil {
		return store.Map(err)
	}
	h.eventNext++
	return nil
}

func errorJSON(err error) json.RawMessage {
	kind := apperror.KindUnavailable
	code := "INTERNAL"
	message := err.Error()
	var ae *apperror.Error
	if asAppError(err, &ae) {
		kind = ae.Kind
		code = ae.Code
		message = ae.Message
	}
	return protocol.RawJSON(protocol.ErrorResponse{
		Kind: string(kind), Code: code, Message: message,
	})
}

func asAppError(err error, target **apperror.Error) bool {
	if e, ok := err.(*apperror.Error); ok {
		*target = e
		return true
	}
	return false
}

func marshalSummary(s *compute.Summary) (json.RawMessage, error) {
	if err := compute.ValidateSummary(s); err != nil {
		return nil, err
	}
	return json.Marshal(protocol.ToSummaryDTO(s))
}

func unmarshalSummary(raw json.RawMessage) (*compute.Summary, error) {
	var dto protocol.SummaryDTO
	if err := json.Unmarshal(raw, &dto); err != nil {
		return nil, apperror.Wrap(apperror.KindUnavailable, "STORED_SUMMARY_INVALID",
			"stored summary cannot be decoded", err)
	}
	return protocol.FromSummaryDTO(dto), nil
}
