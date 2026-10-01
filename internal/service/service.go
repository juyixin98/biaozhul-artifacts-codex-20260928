package service

import (
	"context"
	"strings"

	"diststats/internal/protocol"
	"diststats/internal/stats"
)

type Store interface {
	CreateJob(ctx context.Context, request protocol.JobRequest) (protocol.Job, *protocol.Summary, error)
	GetJob(ctx context.Context, id string) (protocol.Job, *protocol.Summary, error)
	SubmitShard(ctx context.Context, jobID string, shard protocol.ShardRequest, summary *protocol.Summary) (protocol.ShardResult, *protocol.Summary, *protocol.Summary, error)
	RecordRejection(ctx context.Context, jobID, shardID, eventType string, class protocol.ErrorClass, code, detail string, shard *protocol.ShardRequest) (protocol.Event, error)
	ListEvents(ctx context.Context, jobID string) ([]protocol.Event, error)
}

type Service struct {
	store Store
}

func New(store Store) *Service {
	return &Service{store: store}
}

func (s *Service) CreateJob(ctx context.Context, request protocol.JobRequest) (protocol.JobResult, error) {
	if strings.TrimSpace(request.RunID) == "" {
		return protocol.JobResult{}, protocol.NewServiceError(protocol.ClassInput, "MISSING_RUN_ID",
			"run_id is required and must be non-empty", nil)
	}
	if request.Dimension <= 0 || request.Dimension > 64 {
		return protocol.JobResult{}, protocol.NewServiceError(protocol.ClassInput, "INVALID_DIMENSION",
			"dimension must be between 1 and 64", nil)
	}
	if request.Estimator == "" {
		request.Estimator = protocol.Population
	}
	if request.Estimator != protocol.Population && request.Estimator != protocol.Sample {
		return protocol.JobResult{}, protocol.NewServiceError(protocol.ClassInput, "INVALID_ESTIMATOR",
			`estimator must be "population" or "sample"`, nil)
	}
	if request.ExpectedShards < 0 || request.ExpectedShards > 100000 {
		return protocol.JobResult{}, protocol.NewServiceError(protocol.ClassInput, "INVALID_EXPECTED_SHARDS",
			"expected_shards must be between 0 and 100000", nil)
	}

	job, summary, err := s.store.CreateJob(ctx, request)
	if err != nil {
		return protocol.JobResult{}, err
	}
	view := summary.View(job.Estimator)
	return protocol.JobResult{Job: job, Summary: &view}, nil
}

func (s *Service) GetJob(ctx context.Context, id string) (protocol.JobResult, error) {
	job, summary, err := s.store.GetJob(ctx, id)
	if err != nil {
		return protocol.JobResult{}, err
	}
	view := summary.View(job.Estimator)
	return protocol.JobResult{Job: job, Summary: &view}, nil
}

func (s *Service) SubmitShard(ctx context.Context, jobID string, request protocol.ShardRequest) (protocol.ShardResult, error) {
	if strings.TrimSpace(request.ShardID) == "" {
		return protocol.ShardResult{}, protocol.NewServiceError(protocol.ClassInput, "MISSING_SHARD_ID",
			"shard_id is required", nil)
	}

	job, _, err := s.store.GetJob(ctx, jobID)
	if err != nil {
		return protocol.ShardResult{}, err
	}
	summary, err := stats.FromRows(request.Values, request.Weights, job.Dimension)
	if err != nil {
		s.recordFailure(ctx, job.ID, request.ShardID, err, &request)
		return protocol.ShardResult{}, err
	}
	result, _, _, err := s.store.SubmitShard(ctx, job.ID, request, summary)
	if err != nil {
		// A dimension mismatch from storage is impossible here because the
		// job dimensions the summary; retaining the event still aids replay.
		s.recordFailure(ctx, job.ID, request.ShardID, err, &request)
		return protocol.ShardResult{}, err
	}
	return result, nil
}

func (s *Service) Replay(ctx context.Context, jobID string) ([]protocol.Event, protocol.JobResult, error) {
	events, err := s.store.ListEvents(ctx, jobID)
	if err != nil {
		return nil, protocol.JobResult{}, err
	}
	result, err := s.GetJob(ctx, jobID)
	if err != nil {
		return nil, protocol.JobResult{}, err
	}
	return events, result, nil
}

func (s *Service) recordFailure(ctx context.Context, jobID, shardID string, err error, shard *protocol.ShardRequest) {
	serviceErr, ok := protocol.AsServiceError(err)
	if !ok {
		serviceErr = protocol.NewServiceError(protocol.ClassComputation, "COMPUTATION_FAILED", err.Error(), err)
	}
	eventType := "shard_rejected"
	if serviceErr.Class == protocol.ClassResourceExhausted || serviceErr.Class == protocol.ClassComputation {
		eventType = "shard_failed"
	}
	_, _ = s.store.RecordRejection(ctx, jobID, shardID, eventType, serviceErr.Class, serviceErr.Code, serviceErr.Message, shard)
}
