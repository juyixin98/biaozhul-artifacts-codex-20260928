package service

import (
	"context"
	"encoding/json"

	"diststats/internal/apperror"
	"diststats/internal/protocol"
	"diststats/internal/store"
)

func (s *Service) GetRun(ctx context.Context, number int64) (*protocol.RunResponse, error) {
	if number <= 0 {
		return nil, apperror.InvalidInput("INVALID_RUN_NUMBER",
			"run number must be positive")
	}
	run, events, err := s.db.GetRun(ctx, number)
	if err != nil {
		return nil, err
	}
	return runResponse(run, events)
}

func (s *Service) ListJobRuns(ctx context.Context, jobID, excludeRunID string, limit int) ([]protocol.RunResponse, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	runs, err := s.db.ListRuns(ctx, jobID, excludeRunID, limit)
	if err != nil {
		return nil, err
	}
	out := make([]protocol.RunResponse, 0, len(runs))
	for _, run := range runs {
		events, err := s.db.ListRunEvents(ctx, run.ID)
		if err != nil {
			return nil, err
		}
		resp, err := runResponse(run, events)
		if err != nil {
			return nil, err
		}
		out = append(out, *resp)
	}
	return out, nil
}

func runResponse(run store.Run, events []store.RunEvent) (*protocol.RunResponse, error) {
	resp := &protocol.RunResponse{
		ID: run.ID, Number: run.RunNumber, Route: run.Route,
		Method: run.Method, StatusCode: run.StatusCode,
		Status: run.Status, StartedAt: run.StartedAt,
		FinishedAt: run.FinishedAt,
		Events:     make([]protocol.RunEventResponse, 0, len(events)),
	}
	if run.JobID != nil {
		resp.JobID = *run.JobID
	}
	if len(run.Request) > 0 {
		var v any
		if err := json.Unmarshal(run.Request, &v); err != nil {
			return nil, apperror.Wrap(apperror.KindUnavailable, "STORED_REQUEST_INVALID",
				"stored run request cannot be decoded", err)
		}
		resp.Request = v
	}
	if len(run.Error) > 0 {
		var e protocol.ErrorResponse
		if err := json.Unmarshal(run.Error, &e); err != nil {
			return nil, apperror.Wrap(apperror.KindUnavailable, "STORED_ERROR_INVALID",
				"stored run error cannot be decoded", err)
		}
		resp.Error = &e
	}
	for _, event := range events {
		var state any
		if err := json.Unmarshal(event.State, &state); err != nil {
			return nil, apperror.Wrap(apperror.KindUnavailable, "STORED_EVENT_INVALID",
				"stored run event cannot be decoded", err)
		}
		resp.Events = append(resp.Events, protocol.RunEventResponse{
			Index: event.Index, Stage: event.Stage,
			Result: event.Result, Reason: event.Reason,
			State: state, CreatedAt: event.CreatedAt,
		})
	}
	return resp, nil
}
