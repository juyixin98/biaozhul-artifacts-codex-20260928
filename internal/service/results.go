package service

import (
	"context"

	"diststats/internal/compute"
	"diststats/internal/protocol"
)

func (s *Service) GetResult(ctx context.Context, jobID string) (*protocol.ResultResponse, error) {
	job, err := s.db.GetJob(ctx, jobID)
	if err != nil {
		return nil, err
	}
	shards, err := s.db.ListShards(ctx, jobID)
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
	metrics, err := compute.MetricsFromSummary(sum)
	if err != nil {
		return nil, err
	}
	m := protocol.ToMetricsDTO(metrics)
	return &protocol.ResultResponse{
		JobID:       jobID,
		Finalized:   job.Status == "finalized",
		ShardCount:  len(shards),
		ReplayOrder: "current_state",
		Summary:     protocol.ToSummaryDTO(sum),
		Metrics:     &m,
	}, nil
}
