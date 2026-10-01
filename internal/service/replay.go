package service

import (
	"context"

	"diststats/internal/apperror"
	"diststats/internal/compute"
	"diststats/internal/protocol"
)

type ReplayCheckpoint struct {
	Step    int                 `json:"step"`
	ShardID string              `json:"shard_id"`
	Summary protocol.SummaryDTO `json:"summary"`
}

type ReplayReport struct {
	Result      protocol.ResultResponse `json:"result"`
	Checkpoints []ReplayCheckpoint      `json:"checkpoints"`
	Comparison  *Comparison             `json:"comparison,omitempty"`
}

type Comparison struct {
	ReferenceOrder string  `json:"reference_order"`
	MaxMeanError   float64 `json:"max_mean_error"`
	MaxM2Error     float64 `json:"max_m2_error"`
	MaxWeightError float64 `json:"max_weight_error"`
	Tolerance      float64 `json:"tolerance"`
	Accepted       bool    `json:"accepted"`
	Reason         string  `json:"reason"`
}

func (s *Service) Replay(ctx context.Context, run *RunHandle, jobID, order string, compare bool) (*ReplayReport, error) {
	if order != "sequential" && order != "balanced" {
		return nil, apperror.InvalidInput("INVALID_REPLAY_ORDER",
			"replay order must be sequential or balanced")
	}
	job, err := s.db.GetJob(ctx, jobID)
	if err != nil {
		return nil, err
	}
	shards, err := s.db.ListShards(ctx, jobID)
	if err != nil {
		return nil, err
	}
	summaries := make([]*compute.Summary, 0, len(shards))
	checkpoints := make([]ReplayCheckpoint, 0, len(shards)+1)
	if order == "sequential" {
		acc := compute.Empty(job.Dim)
		for i, sh := range shards {
			sum, err := unmarshalSummary(sh.Summary)
			if err != nil {
				return nil, err
			}
			if err := compute.MergeInto(acc, sum); err != nil {
				return nil, err
			}
			checkpoints = append(checkpoints, ReplayCheckpoint{
				Step: i + 1, ShardID: sh.ShardID,
				Summary: protocol.ToSummaryDTO(acc),
			})
		}
		summaries = []*compute.Summary{acc}
	} else {
		for _, sh := range shards {
			sum, err := unmarshalSummary(sh.Summary)
			if err != nil {
				return nil, err
			}
			summaries = append(summaries, sum)
		}
	}

	replayed, balancedCheckpoints, err := compute.MergeTreeWithCheckpoints(summaries, order == "balanced")
	if err != nil {
		return nil, err
	}
	if replayed == nil {
		replayed = compute.Empty(job.Dim)
	}
	if order == "balanced" {
		for step, cp := range balancedCheckpoints {
			checkpoints = append(checkpoints, ReplayCheckpoint{
				Step: step + 1, ShardID: cp.ShardID,
				Summary: protocol.ToSummaryDTO(cp.Summary),
			})
		}
	}
	metrics, err := compute.MetricsFromSummary(replayed)
	if err != nil {
		return nil, err
	}
	var dtoMetrics *protocol.MetricsDTO
	if metrics != nil {
		m := protocol.ToMetricsDTO(metrics)
		dtoMetrics = &m
	}

	var comparison *Comparison
	if compare {
		other := "sequential"
		if order == "sequential" {
			other = "balanced"
		}
		otherShardSummaries := make([]*compute.Summary, 0, len(shards))
		for _, sh := range shards {
			sum, err := unmarshalSummary(sh.Summary)
			if err != nil {
				return nil, err
			}
			otherShardSummaries = append(otherShardSummaries, sum)
		}
		reference, err := compute.MergeTree(otherShardSummaries, other == "balanced")
		if err != nil {
			return nil, err
		}
		if reference == nil {
			reference = compute.Empty(job.Dim)
		}
		comparison = compareSummaries(replayed, reference, other)
		if err := s.AddEvent(ctx, run, "replay_tree_compared", "succeeded",
			"replayed stored partitions through two merge trees",
			comparison); err != nil {
			return nil, err
		}
	}
	if err := s.AddEvent(ctx, run, "partition_replay", "succeeded",
		"replayed all persisted partitions without mutating job state",
		map[string]any{"order": order, "checkpoints": len(checkpoints)}); err != nil {
		return nil, err
	}

	return &ReplayReport{
		Result: protocol.ResultResponse{
			JobID: jobID, Finalized: job.Status == "finalized",
			ShardCount: len(shards), ReplayOrder: order,
			Summary: protocol.ToSummaryDTO(replayed), Metrics: dtoMetrics,
		},
		Checkpoints: checkpoints,
		Comparison:  comparison,
	}, nil
}

func compareSummaries(got, ref *compute.Summary, referenceOrder string) *Comparison {
	const rel = 1e-10
	const abs = 1e-8
	maxMean := maxAbsDiff(got.AbsoluteMeans(), ref.AbsoluteMeans())
	maxM2 := maxAbsDiff(effectiveM2(got), effectiveM2(ref))
	maxWeight := absFloat(got.Weight - ref.Weight)
	refMean := ref.AbsoluteMeans()
	refM2 := effectiveM2(ref)
	tol := maxTolerance(abs, rel, refMean, refM2)
	accepted := got.Dim == ref.Dim &&
		got.N == ref.N &&
		maxWeight <= abs &&
		maxMean <= maxTolerance(abs, rel, refMean) &&
		maxM2 <= tol
	reason := "differences are within configured non-bitwise float64 tolerance"
	if !accepted {
		reason = "merge tree result exceeded configured tolerance"
	}
	return &Comparison{
		ReferenceOrder: referenceOrder,
		MaxMeanError:   maxMean,
		MaxM2Error:     maxM2,
		MaxWeightError: maxWeight,
		Tolerance:      tol,
		Accepted:       accepted,
		Reason:         reason,
	}
}

func effectiveM2(s *compute.Summary) []float64 {
	out := make([]float64, len(s.M2))
	for i := range out {
		out[i] = s.M2[i] + s.M2Comp[i]
	}
	return out
}

func maxAbsDiff(a, b []float64) float64 {
	var m float64
	for i := range a {
		m = maxFloat(m, absFloat(a[i]-b[i]))
	}
	return m
}

func maxTolerance(abs, rel float64, groups ...[]float64) float64 {
	m := abs
	for _, g := range groups {
		for _, v := range g {
			m = maxFloat(m, abs+rel*absFloat(v))
		}
	}
	return m
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
