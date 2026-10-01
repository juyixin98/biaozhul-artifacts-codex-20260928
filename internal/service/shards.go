package service

import (
	"context"
	"database/sql"
	"math"
	"strings"

	"diststats/internal/apperror"
	"diststats/internal/compute"
	"diststats/internal/protocol"
	"diststats/internal/store"
)

func (s *Service) SubmitShard(ctx context.Context, run *RunHandle, jobID string, req protocol.SubmitShardRequest) (*protocol.ShardResponse, error) {
	if strings.TrimSpace(req.ShardID) == "" {
		return nil, apperror.InvalidInput("MISSING_SHARD_ID", "shard_id is required")
	}
	if len(req.Observations) > 0 && req.Summary != nil {
		return nil, apperror.InvalidInput("AMBIGUOUS_SHARD_PAYLOAD",
			"provide either observations or summary, not both")
	}
	if len(req.Observations) > s.limits.MaxObservations {
		return nil, apperror.ResourceExhausted("OBSERVATION_LIMIT_EXCEEDED",
			"shard observation count exceeds configured maximum")
	}

	var resp *protocol.ShardResponse
	err := s.db.InTx(ctx, func(tx *sql.Tx) error {
		job, err := store.GetJobForUpdate(ctx, tx, jobID)
		if err != nil {
			return err
		}
		if job.Status == "finalized" {
			return apperror.Conflict("JOB_FINALIZED",
				"finalized jobs reject new shard submissions")
		}

		shard, err := buildShard(req, job.Dim)
		if err != nil {
			return err
		}
		before, err := unmarshalSummary(job.Aggregate)
		if err != nil {
			return err
		}

		seq, err := store.NextShardSeq(ctx, tx, jobID)
		if err != nil {
			return err
		}
		raw, err := marshalSummary(shard)
		if err != nil {
			return err
		}
		if err := store.InsertShard(ctx, tx, store.Shard{
			JobID: jobID, ShardID: req.ShardID, Seq: seq,
			N: shard.N, Weight: shard.Weight, Summary: raw,
			SubmittedRunID: run.ID,
		}); err != nil {
			return err
		}
		if err := addTxEvent(ctx, tx, run, "shard_deduplicated", "info",
			"primary key (job_id, shard_id) protects this insert against duplicate merge",
			EventState{ShardID: req.ShardID, Seq: seq}); err != nil {
			return err
		}

		after := compute.Clone(before)
		if err := compute.MergeInto(after, shard); err != nil {
			return err
		}
		afterRaw, err := marshalSummary(after)
		if err != nil {
			return err
		}
		newVersion := job.Version + 1
		if err := store.UpdateJobAggregate(ctx, tx, jobID, afterRaw, newVersion); err != nil {
			return err
		}
		if err := addTxEvent(ctx, tx, run, "stable_summary_merged", "succeeded",
			"merged weighted Chan/Welford summary and advanced job version",
			EventState{
				Before:  protocol.ToSummaryDTO(before),
				After:   protocol.ToSummaryDTO(after),
				Shard:   dtoPtr(protocol.ToSummaryDTO(shard)),
				ShardID: req.ShardID,
				Seq:     seq,
			}); err != nil {
			return err
		}

		stored, err := store.GetShard(ctx, tx, jobID, req.ShardID)
		if err != nil {
			return err
		}
		resp = &protocol.ShardResponse{
			JobID: jobID, ShardID: req.ShardID, Seq: seq,
			RunID: run.ID, RunNumber: run.Number,
			CreatedAt: stored.CreatedAt, JobVersion: newVersion,
			Summary:    protocol.ToSummaryDTO(shard),
			JobSummary: protocol.ToSummaryDTO(after),
		}
		return nil
	})
	if err != nil {
		_ = s.AddEvent(ctx, run, "shard_submit_rejected", "failed",
			err.Error(), map[string]any{
				"shard_id": req.ShardID,
			})
		return nil, err
	}
	return resp, nil
}

func buildShard(req protocol.SubmitShardRequest, expectedDim int) (*compute.Summary, error) {
	if req.Summary != nil {
		s := protocol.FromSummaryDTO(*req.Summary)
		if err := compute.ValidateSummary(s); err != nil {
			return nil, err
		}
		if s.Dim != expectedDim {
			return nil, apperror.InvalidInput("DIMENSION_MISMATCH",
				"shard summary dimension does not match job dimension")
		}
		return s, nil
	}

	if len(req.Origin) > 0 && len(req.Origin) != expectedDim {
		return nil, apperror.InvalidInput("DIMENSION_MISMATCH",
			"origin dimension does not match job dimension")
	}
	for _, v := range req.Origin {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, apperror.InvalidInput("INVALID_ORIGIN",
				"all origin coordinates must be finite")
		}
	}
	s := compute.EmptyAt(expectedDim, req.Origin)
	for _, o := range req.Observations {
		w := 1.0
		if o.Weight != nil {
			w = *o.Weight
		}
		if len(o.Values) != expectedDim {
			return nil, apperror.InvalidInput("DIMENSION_MISMATCH",
				"observation dimension does not match job dimension")
		}
		if err := compute.AddPoint(s, o.Values, w); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func dtoPtr(v protocol.SummaryDTO) *protocol.SummaryDTO { return &v }
