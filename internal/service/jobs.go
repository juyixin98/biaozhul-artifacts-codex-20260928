package service

import (
	"context"
	"time"

	"diststats/internal/apperror"
	"diststats/internal/compute"
	"diststats/internal/id"
	"diststats/internal/protocol"
	"diststats/internal/store"
)

func (s *Service) CreateJob(ctx context.Context, run *RunHandle, req protocol.CreateJobRequest) (*store.Job, error) {
	if req.Dimension <= 0 {
		return nil, apperror.InvalidInput("INVALID_DIMENSION",
			"dimension must be a positive integer")
	}
	if req.Dimension > s.limits.MaxDimensions {
		return nil, apperror.ResourceExhausted("DIMENSION_LIMIT_EXCEEDED",
			"requested dimension exceeds configured maximum")
	}
	jobID, err := id.New()
	if err != nil {
		return nil, err
	}
	empty := compute.Empty(req.Dimension)
	raw, err := marshalSummary(empty)
	if err != nil {
		return nil, err
	}
	job := store.Job{
		ID:        jobID,
		Dim:       req.Dimension,
		Status:    "open",
		Aggregate: raw,
		Version:   0,
		CreatedAt: time.Now().UTC(),
	}
	if err := s.db.CreateJob(ctx, job); err != nil {
		return nil, err
	}
	if err := s.db.AttachRunJob(ctx, run.ID, jobID); err != nil {
		return nil, err
	}
	if err := s.AddEvent(ctx, run, "job_created", "succeeded",
		"validated dimension and initialized empty stable accumulator", EventState{
			After: protocol.ToSummaryDTO(empty), Dimension: req.Dimension,
		}); err != nil {
		return nil, err
	}
	return &job, nil
}

func (s *Service) GetJobResponse(ctx context.Context, jobID string) (*protocol.JobResponse, error) {
	job, err := s.db.GetJob(ctx, jobID)
	if err != nil {
		return nil, err
	}
	sum, err := unmarshalSummary(job.Aggregate)
	if err != nil {
		return nil, err
	}
	if err := compute.ValidateSummary(sum); err != nil {
		return nil, err
	}
	return jobResponse(job, sum), nil
}

func jobResponse(job store.Job, sum *compute.Summary) *protocol.JobResponse {
	var finalized *time.Time
	if job.FinalizedAt != nil {
		t := *job.FinalizedAt
		finalized = &t
	}
	return &protocol.JobResponse{
		ID: job.ID, Dimension: job.Dim, Status: job.Status,
		Version: job.Version, CreatedAt: job.CreatedAt,
		FinalizedAt: finalized, Summary: protocol.ToSummaryDTO(sum),
	}
}

func (s *Service) Finalize(ctx context.Context, run *RunHandle, jobID string) (*store.Job, error) {
	job, err := s.db.GetJob(ctx, jobID)
	if err != nil {
		return nil, err
	}
	if job.Status == "finalized" {
		err := apperror.Conflict("JOB_FINALIZED",
			"job is already finalized and rejects new state changes")
		_ = s.AddEvent(ctx, run, "finalize_rejected", "failed",
			err.Error(), map[string]any{"job_id": jobID})
		return nil, err
	}
	before, err := unmarshalSummary(job.Aggregate)
	if err != nil {
		return nil, err
	}
	if err := s.AddEvent(ctx, run, "finalize_validate", "info",
		"job is open and eligible for finalization", EventState{
			Before: protocol.ToSummaryDTO(before),
		}); err != nil {
		return nil, err
	}
	at := time.Now().UTC()
	updated, err := s.db.FinalizeJob(ctx, jobID, at)
	if err != nil {
		_ = s.AddEvent(ctx, run, "finalize_rejected", "failed",
			err.Error(), map[string]any{"job_id": jobID})
		return nil, err
	}
	if err := s.AddEvent(ctx, run, "job_finalized", "succeeded",
		"state changed from open to finalized", EventState{
			Before: protocol.ToSummaryDTO(before),
			After:  protocol.ToSummaryDTO(before),
		}); err != nil {
		return nil, err
	}
	return &updated, nil
}
