//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"opp300-speculative-batch/internal/protocol"
)

func TestCoordinatorRestartFencesOldAttemptAndReplaysJobToCompletion(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, databaseURL(t))
	fixture := loadRestartFixture(t)

	job := h.submit(ctx, protocol.JobSpec{
		IDempotencyKey: unique(fixture.IDempotencyKey),
		Shards:         []protocol.ShardInput{fixture.Shard},
		Policy: protocol.Policy{
			MaxAttemptsPerShard: 2,
			MaxActiveAttempts:   4,
			LeaseMS:             60_000,
			SpeculateAfterMS:    10_000,
			HeartbeatIntervalMS: 30,
		},
	})

	waitForAttemptState(t, ctx, h, job.ID, 1, protocol.AttemptRunning)
	oldEpoch := h.epoch
	h.restartCoordinator(ctx)

	finished := h.waitJobState(ctx, job.ID, protocol.JobSucceeded)
	attempts := h.attempts(ctx, job.ID)
	events := h.events(ctx, job.ID)
	merge := h.result(ctx, job.ID)

	if len(attempts) != 2 {
		t.Fatalf("attempts after restart = %d, want old interrupted + replacement original: %+v", len(attempts), attempts)
	}
	if attempts[0].State != protocol.AttemptInterrupted || attempts[0].Epoch != oldEpoch {
		t.Fatalf("old attempt = %+v, want interrupted in old epoch", attempts[0])
	}
	if attempts[1].Kind != "original" || attempts[1].State != protocol.AttemptSucceeded || attempts[1].Epoch == oldEpoch {
		t.Fatalf("new attempt = %+v, want replacement original in a fresh epoch", attempts[1])
	}
	if len(merge.Shards) != 1 || merge.Shards[0].Output != fixture.ExpectedOutput {
		t.Fatalf("merge = %+v, want one fixture output %q", merge, fixture.ExpectedOutput)
	}
	assertEventType(t, events, "COORDINATOR_EPOCH_FENCED")
	assertEventType(t, events, "SHARD_RETRY")
	assertEventType(t, events, "JOB_SUCCEEDED")
	t.Logf("fixture=%s run=%s version=%s old_epoch=%s new_epoch=%s job=%s final_version=%d",
		fixture.InputRunID, fixture.IDempotencyKey, fixture.Version,
		oldEpoch, h.epoch, job.ID, finished.Version)
}

func TestReplayUsesStoredInputsAndIndependentRun(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, databaseURL(t))
	fixture := loadSpeculativeFixture(t)

	source := h.submit(ctx, protocol.JobSpec{
		IDempotencyKey: unique(fixture.IDempotencyKey),
		Shards:         []protocol.ShardInput{fixture.Shard},
		Policy: protocol.Policy{
			MaxAttemptsPerShard: 2,
			MaxActiveAttempts:   4,
			LeaseMS:             10_000,
			SpeculateAfterMS:    100,
			HeartbeatIntervalMS: 30,
		},
	})
	h.waitJobState(ctx, source.ID, protocol.JobSucceeded)

	var envelope struct {
		Job protocol.Job `json:"job"`
	}
	response := h.doJSON(ctx, http.MethodPost,
		h.coordinatorURL+"/v1/jobs/"+source.ID+"/replay",
		map[string]any{"idempotency_key": unique("replay-" + fixture.IDempotencyKey)})
	defer response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		t.Fatalf("replay status %d", response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(&envelope); err != nil {
		t.Fatal(err)
	}
	replayed := h.waitJobState(ctx, envelope.Job.ID, protocol.JobSucceeded)
	if replayed.ID == source.ID {
		t.Fatal("replay rewrote source job")
	}
	merge := h.result(ctx, replayed.ID)
	if len(merge.Shards) != 1 || merge.Shards[0].Output != fixture.ExpectedOutput {
		t.Fatalf("replayed merge = %+v, want one independent result", merge)
	}
}
