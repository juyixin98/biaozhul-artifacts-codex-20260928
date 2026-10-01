package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"specbatch/internal/kernel"
	"specbatch/internal/logging"
	"specbatch/internal/model"
)

type Worker struct {
	ID          string
	Client      *Client
	StagingRoot string
	Log         *logging.Logger
	PollEvery   time.Duration
}

type Behavior struct {
	DelayMS     int
	Fail        bool
	FailMessage string
}

func (w *Worker) Run(ctx context.Context) error {
	if w.PollEvery <= 0 {
		w.PollEvery = 20 * time.Millisecond
	}
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}
		lease, ok, err := w.Client.Poll(ctx, w.ID)
		if err != nil {
			if errors.Is(ctx.Err(), context.Canceled) {
				return nil
			}
			w.Log.Error("poll_failed", map[string]any{"worker_id": w.ID, "error": err.Error()})
			if !sleep(ctx, 100*time.Millisecond) {
				return nil
			}
			continue
		}
		if !ok {
			if !sleep(ctx, w.PollEvery) {
				return nil
			}
			continue
		}
		retry, err := w.execute(ctx, lease)
		if err != nil {
			w.Log.Error("execute_failed", map[string]any{
				"worker_id": w.ID, "job_id": lease.JobID, "shard_id": lease.ShardID,
				"attempt_no": lease.AttemptNo, "error": err.Error(),
			})
		}
		if !retry {
			// Continue polling: a worker process is a reusable participant.
		}
	}
}

func (w *Worker) execute(ctx context.Context, lease model.Lease) (bool, error) {
	b := behavior(lease)
	runID := fmt.Sprintf("%s-%s-a%d-%s", lease.JobID, lease.ShardID, lease.AttemptNo, w.ID)
	dir := filepath.Join(w.StagingRoot, runID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return false, w.reportFailure(ctx, lease, model.FailureUnknown, err.Error())
	}
	defer func() {
		if err := os.RemoveAll(dir); err != nil {
			w.Log.Error("staging_cleanup_failed", map[string]any{"dir": dir, "error": err.Error()})
		}
	}()

	w.Log.Info("attempt_started", map[string]any{
		"run_id": runID, "worker_id": w.ID, "job_id": lease.JobID, "shard_id": lease.ShardID,
		"attempt_no": lease.AttemptNo, "kind": lease.Kind, "kernel": lease.Kernel,
		"kernel_version": lease.KernelVersion, "records": len(lease.Records),
		"record_ids": recordIDs(lease.Records), "synthetic_delay_ms": b.DelayMS, "synthetic_fail": b.Fail,
	})
	_ = w.heartbeat(ctx, lease, 0.05)

	// Heartbeats during the slow section give the coordinator explainable progress.
	delay := time.Duration(b.DelayMS) * time.Millisecond
	steps := 4
	if steps > 1 && delay > 0 {
		for i := 1; i <= steps; i++ {
			if !sleep(ctx, delay/time.Duration(steps)) {
				return false, ctx.Err()
			}
			_ = w.heartbeat(ctx, lease, 0.05+0.15*float64(i)/float64(steps))
		}
	} else {
		if !sleep(ctx, delay) {
			return false, ctx.Err()
		}
	}
	if b.Fail {
		category := model.FailureSimulatedWorker
		msg := fmt.Sprintf("synthetic failure for attempt %d: %s", lease.AttemptNo, b.FailMessage)
		w.Log.Info("synthetic_failure", map[string]any{
			"run_id": runID, "job_id": lease.JobID, "shard_id": lease.ShardID,
			"attempt_no": lease.AttemptNo, "failure_category": category, "step": "post_delay_before_compute",
		})
		return false, w.reportFailure(ctx, lease, category, msg)
	}
	_ = w.heartbeat(ctx, lease, 0.75)
	doc, err := kernel.CanonicalArtifact(lease)
	if err != nil {
		return false, w.reportFailure(ctx, lease, model.FailureValidation, err.Error())
	}
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return false, w.reportFailure(ctx, lease, model.FailureUnknown, err.Error())
	}
	tmp := filepath.Join(dir, "artifact.json")
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return false, w.reportFailure(ctx, lease, model.FailureUnknown, err.Error())
	}
	w.Log.Info("compute_step_completed", map[string]any{
		"run_id": runID, "job_id": lease.JobID, "shard_id": lease.ShardID, "attempt_no": lease.AttemptNo,
		"step": "canonical_artifact_written", "bytes": len(raw), "temporary_path": tmp,
	})
	_ = w.heartbeat(ctx, lease, 0.95)
	err = w.Client.Complete(ctx, lease.Token, w.ID, raw)
	if err != nil {
		w.Log.Error("late_submission_rejected_or_transport_failed", map[string]any{
			"run_id": runID, "job_id": lease.JobID, "shard_id": lease.ShardID,
			"attempt_no": lease.AttemptNo, "error": err.Error(),
		})
		return false, nil
	}
	w.Log.Info("submission_accepted", map[string]any{
		"run_id": runID, "job_id": lease.JobID, "shard_id": lease.ShardID, "attempt_no": lease.AttemptNo,
	})
	return true, nil
}

func behavior(lease model.Lease) Behavior {
	key := strconv.Itoa(lease.AttemptNo)
	delay := 0
	fail := false
	if lease.Behavior != nil {
		if v, ok := lease.Behavior[key+"_delay_ms"]; ok {
			delay = v
		}
		if v, ok := lease.Behavior[key+"_fail"]; ok && v == 1 {
			fail = true
		}
	}
	failKinds := map[string]bool{}
	for _, kind := range lease.FailKinds {
		failKinds[kind] = true
	}
	if failKinds[key] {
		fail = true
	}
	return Behavior{DelayMS: delay, Fail: fail, FailMessage: "fixture-injected"}
}

func (w *Worker) heartbeat(ctx context.Context, lease model.Lease, progress float64) error {
	err := w.Client.Heartbeat(ctx, lease.Token, w.ID, progress)
	if err != nil {
		w.Log.Error("heartbeat_rejected", map[string]any{
			"worker_id": w.ID, "job_id": lease.JobID, "shard_id": lease.ShardID,
			"attempt_no": lease.AttemptNo, "progress": progress, "error": err.Error(),
		})
		return err
	}
	w.Log.Debug("heartbeat", map[string]any{
		"worker_id": w.ID, "job_id": lease.JobID, "shard_id": lease.ShardID,
		"attempt_no": lease.AttemptNo, "progress": progress,
	})
	return nil
}

func (w *Worker) reportFailure(ctx context.Context, lease model.Lease, category, message string) error {
	err := w.Client.Fail(ctx, lease.Token, w.ID, category, message)
	if err != nil {
		w.Log.Error("failure_report_rejected", map[string]any{
			"worker_id": w.ID, "job_id": lease.JobID, "shard_id": lease.ShardID,
			"attempt_no": lease.AttemptNo, "failure_category": category, "error": err.Error(),
		})
		return err
	}
	w.Log.Info("failure_reported", map[string]any{
		"worker_id": w.ID, "job_id": lease.JobID, "shard_id": lease.ShardID,
		"attempt_no": lease.AttemptNo, "failure_category": category, "message": message,
	})
	return nil
}

func sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

func recordIDs(records []model.InputRecord) []string {
	out := make([]string, 0, len(records))
	for _, r := range records {
		out = append(out, r.ID)
	}
	return out
}
