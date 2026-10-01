package replay

import (
	"context"
	"fmt"

	"opp299/internal/kernel"
	"opp299/internal/protocol"
	"opp299/internal/source"
)

type Report struct {
	CheckpointID    int64                        `json:"checkpoint_id"`
	Mode            protocol.Mode                `json:"mode"`
	Positions       map[string]protocol.Position `json:"positions"`
	Inflight        []protocol.Record            `json:"inflight"`
	ReplayedOutputs []protocol.Output            `json:"replayed_outputs"`
	BaselineState   *protocol.State              `json:"baseline_state"`
	PostReplayState *protocol.State              `json:"post_replay_state"`
}

// Inflight restores a runner from cp and applies exactly the records named by
// cp.InFlight. It does not inspect queue or live state, so unaligned recovery
// cannot double-count records already included in the snapshot.
func Inflight(ctx context.Context, cfg protocol.Config, store kernel.Store, cp protocol.Checkpoint) (*kernel.Runner, Report, error) {
	runner, err := kernel.RestoreFromCheckpoint(cfg, store, cp)
	if err != nil {
		return nil, Report{}, err
	}
	report := Report{
		CheckpointID:  cp.ID,
		Mode:          cp.Mode,
		Positions:     cp.Positions,
		Inflight:      source.InflightOrder(cp),
		BaselineState: cp.State.Clone(),
	}
	for _, rec := range report.Inflight {
		select {
		case <-ctx.Done():
			return nil, report, &kernel.ClassifiedError{Category: kernel.CategoryCancelled, Err: ctx.Err()}
		default:
		}
		out := runner.ApplyRecoveryRecord(rec)
		report.ReplayedOutputs = append(report.ReplayedOutputs, out)
	}
	report.PostReplayState = runner.StateSnapshot()
	if len(report.Inflight) != inflightCount(cp) {
		return nil, report, &kernel.ClassifiedError{
			Category: kernel.CategoryProtocol,
			Err:      fmt.Errorf("replay list changed during recovery"),
		}
	}
	return runner, report, nil
}

func inflightCount(cp protocol.Checkpoint) int {
	total := 0
	for _, items := range cp.InFlight {
		total += len(items)
	}
	return total
}
