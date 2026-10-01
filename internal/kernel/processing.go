package kernel

import (
	"context"
	"fmt"
	"sort"

	"checkpointlab/internal/protocol"
	"checkpointlab/internal/storage"
)

func (o *Operator) processRecord(ctx context.Context, r protocol.Record, requestID string) (Result, error) {
	if err := r.Validate(); err != nil {
		return Result{}, protocol.NewTypedError(protocol.CategoryValidation, err.Error(), false)
	}
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.active != nil && r.Offset < o.active.baseOffsets[r.Stream] {
		return Result{}, protocol.NewTypedError(protocol.CategoryReplayDuplicate,
			fmt.Sprintf("record %s:%d is already included in checkpoint base", r.Stream, r.Offset), false)
	}
	if r.Offset != o.next[r.Stream] {
		category := protocol.CategoryReplayGap
		if r.Offset < o.next[r.Stream] {
			category = protocol.CategoryReplayDuplicate
		}
		return Result{}, protocol.NewTypedError(category,
			fmt.Sprintf("stream %s expected offset %d got %d", r.Stream, o.next[r.Stream], r.Offset), false)
	}

	if o.active != nil && o.cfg.Mode == protocol.ModeAligned {
		if _, blocked := o.active.targets[r.Stream]; blocked {
			return Result{}, protocol.NewTypedError(protocol.CategoryAlignmentBlocked,
				fmt.Sprintf("aligned gate %s is blocked at checkpoint %d", r.Stream, o.active.epoch), true)
		}
	}

	inflight := false
	if o.active != nil && o.cfg.Mode == protocol.ModeUnaligned && r.Offset >= o.active.baseOffsets[r.Stream] {
		target, seen := o.active.targets[r.Stream]
		inflight = !seen || r.Offset < target
	}
	if inflight {
		count := o.inflightCountLocked()
		if count+1 > o.cfg.InflightLimit {
			return Result{}, protocol.NewTypedError(protocol.CategoryBackpressureLimit,
				fmt.Sprintf("unaligned inflight limit %d reached", o.cfg.InflightLimit), true)
		}
		ref := storage.RefFromRecord(r)
		if err := o.store.AppendInflight(ctx, o.cfg.JobID, o.active.epoch, o.active.attemptID, ref); err != nil {
			return Result{}, err
		}
		o.active.records[r.Stream] = append(o.active.records[r.Stream], ref)
		if count+1 > o.peakInflight {
			o.peakInflight = count + 1
		}
	}

	o.state[r.Key] += r.Value
	o.next[r.Stream] = r.Offset + 1
	o.stateVersion++
	o.processed++

	outputID := outputID(o.cfg.JobID, r.Stream, r.Offset)
	committed := o.committedEpoch >= r.Epoch
	if err := o.store.StageOutput(ctx, storage.Output{
		JobID:     o.cfg.JobID,
		Epoch:     r.Epoch,
		Stream:    r.Stream,
		Offset:    r.Offset,
		Key:       r.Key,
		Value:     r.Value,
		Committed: committed,
		OutputID:  outputID,
	}); err != nil {
		return Result{}, err
	}

	activeEpoch := o.activeEpochLocked()
	delta := 0
	if inflight {
		delta = 1
	}
	result := Result{
		Accepted:      true,
		Stream:        r.Stream,
		Offset:        r.Offset,
		Epoch:         r.Epoch,
		Committed:     committed,
		ActiveEpoch:   activeEpoch,
		InflightDelta: delta,
		OutputID:      outputID,
		StateVersion:  o.stateVersion,
		State:         cloneState(o.state),
		Metrics:       o.metricsLocked(),
	}
	_ = o.log(ctx, requestID, r.Epoch, "record_applied", "kernel",
		fmt.Sprintf("stream=%s offset=%d inflight=%t", r.Stream, r.Offset, inflight))
	return result, nil
}

func (o *Operator) processBarrier(ctx context.Context, b protocol.Barrier, requestID string) (Result, error) {
	if err := b.Validate(); err != nil {
		return Result{}, protocol.NewTypedError(protocol.CategoryValidation, err.Error(), false)
	}
	o.mu.Lock()
	defer o.mu.Unlock()

	if b.Epoch <= o.committedEpoch {
		return Result{}, protocol.NewTypedError(protocol.CategoryBarrierOrder,
			fmt.Sprintf("barrier epoch %d is already committed", b.Epoch), false)
	}
	if b.Epoch <= o.lastBarrier[b.Stream] {
		return Result{}, protocol.NewTypedError(protocol.CategoryBarrierOrder,
			fmt.Sprintf("stream %s barrier moved backwards: %d after %d", b.Stream, b.Epoch, o.lastBarrier[b.Stream]), false)
	}
	if o.active != nil && b.Epoch != o.active.epoch {
		return Result{}, protocol.NewTypedError(protocol.CategoryCheckpointScope,
			fmt.Sprintf("checkpoint %d is active; epoch %d cannot start", o.active.epoch, b.Epoch), false)
	}
	if b.Offset != o.next[b.Stream] {
		return Result{}, protocol.NewTypedError(protocol.CategoryBarrierOrder,
			fmt.Sprintf("barrier stream %s offset %d does not match next offset %d", b.Stream, b.Offset, o.next[b.Stream]), false)
	}

	sealed := false
	if o.active == nil {
		if err := o.startCheckpointLocked(ctx, b); err != nil {
			return Result{}, err
		}
		if o.cfg.Mode == protocol.ModeAligned {
			o.gates[b.Stream].Pause()
		}
		o.lastBarrier[b.Stream] = b.Epoch
	} else {
		if _, seen := o.active.targets[b.Stream]; seen {
			return Result{}, protocol.NewTypedError(protocol.CategoryBarrierOrder,
				fmt.Sprintf("duplicate barrier stream=%s epoch=%d", b.Stream, b.Epoch), false)
		}
		if err := o.store.ObserveBarrier(ctx, o.cfg.JobID, b.Epoch, o.active.attemptID, b); err != nil {
			return Result{}, err
		}
		o.active.targets[b.Stream] = b.Offset
		o.lastBarrier[b.Stream] = b.Epoch
		if o.cfg.Mode == protocol.ModeAligned {
			if err := o.sealAlignedLocked(ctx); err != nil {
				return Result{}, err
			}
			o.gates[protocol.StreamA].Resume()
			o.gates[protocol.StreamB].Resume()
		} else {
			if err := o.sealUnalignedLocked(ctx); err != nil {
				return Result{}, err
			}
		}
		sealed = true
	}

	activeEpoch := o.activeEpochLocked()
	result := Result{
		Accepted:     true,
		Stream:       b.Stream,
		Offset:       b.Offset,
		Epoch:        b.Epoch,
		Committed:    sealed,
		ActiveEpoch:  activeEpoch,
		StateVersion: o.stateVersion,
		State:        cloneState(o.state),
		Metrics:      o.metricsLocked(),
	}
	_ = o.log(ctx, requestID, b.Epoch, "barrier_observed", "kernel",
		fmt.Sprintf("stream=%s target_offset=%d sealed=%t", b.Stream, b.Offset, sealed))
	return result, nil
}

func (o *Operator) startCheckpointLocked(ctx context.Context, b protocol.Barrier) error {
	attemptID, err := newAttemptID()
	if err != nil {
		return err
	}
	active := &activeCheckpoint{
		epoch:     b.Epoch,
		attemptID: attemptID,
		base:      cloneState(o.state),
		baseOffsets: map[string]int64{
			protocol.StreamA: o.next[protocol.StreamA],
			protocol.StreamB: o.next[protocol.StreamB],
		},
		targets: map[string]int64{b.Stream: b.Offset},
		records: map[string][]storage.RecordRef{
			protocol.StreamA: {},
			protocol.StreamB: {},
		},
	}
	snap := storage.Snapshot{
		JobID:     o.cfg.JobID,
		Epoch:     b.Epoch,
		AttemptID: attemptID,
		Mode:      o.cfg.Mode,
		Base:      cloneState(active.base),
		Inflights: map[string][]storage.RecordRef{},
		Targets: map[string]storage.BarrierTarget{
			b.Stream: {Offset: b.Offset, Seen: true},
		},
		SeenBarriers: map[string]bool{b.Stream: true},
		BaseOffsets:  cloneInt64Map(active.baseOffsets),
	}
	if err := o.store.StartCheckpoint(ctx, snap); err != nil {
		return err
	}
	o.active = active
	return nil
}

func (o *Operator) sealAlignedLocked(ctx context.Context) error {
	// Both gates were frozen at their own barriers. Every pre-barrier event
	// was processed before its barrier reached the operator, so base is final.
	refs := []storage.RecordRef{}
	if _, err := o.store.SealCheckpoint(ctx, o.cfg.JobID, o.active.epoch, o.active.attemptID, o.active.base, refs); err != nil {
		return err
	}
	o.finishSealLocked()
	return nil
}

func (o *Operator) sealUnalignedLocked(ctx context.Context) error {
	for _, stream := range []string{protocol.StreamA, protocol.StreamB} {
		target := o.active.targets[stream]
		base := o.active.baseOffsets[stream]
		refs := o.active.records[stream]
		sort.Slice(refs, func(i, j int) bool { return refs[i].Offset < refs[j].Offset })
		if len(refs) > 0 {
			if refs[0].Offset != base {
				return protocol.NewTypedError(protocol.CategoryReplayGap,
					fmt.Sprintf("stream %s inflight starts at %d, base ends at %d", stream, refs[0].Offset, base), false)
			}
			if refs[len(refs)-1].Offset != target-1 {
				return protocol.NewTypedError(protocol.CategoryReplayGap,
					fmt.Sprintf("stream %s inflight ends at %d, barrier target is %d", stream, refs[len(refs)-1].Offset, target), false)
			}
			for i := 1; i < len(refs); i++ {
				if refs[i].Offset != refs[i-1].Offset+1 {
					return protocol.NewTypedError(protocol.CategoryReplayGap,
						fmt.Sprintf("stream %s inflight has gap before offset %d", stream, refs[i].Offset), false)
				}
			}
		} else if base != target {
			return protocol.NewTypedError(protocol.CategoryReplayGap,
				fmt.Sprintf("stream %s expected %d inflight records, got 0", stream, target-base), false)
		}
	}
	refs := append(append([]storage.RecordRef{}, o.active.records[protocol.StreamA]...), o.active.records[protocol.StreamB]...)
	snap, err := o.store.SealCheckpoint(ctx, o.cfg.JobID, o.active.epoch, o.active.attemptID, o.active.base, refs)
	if err != nil {
		return err
	}
	if len(snap.Inflights[protocol.StreamA])+len(snap.Inflights[protocol.StreamB]) != len(refs) {
		return protocol.NewTypedError(protocol.CategoryUncertainStorage, "stored inflight count does not match sealing request", true)
	}
	o.finishSealLocked()
	return nil
}

func (o *Operator) finishSealLocked() {
	o.committedEpoch = o.active.epoch
	o.active = nil
	o.sealed++
}

func cloneInt64Map(in map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
