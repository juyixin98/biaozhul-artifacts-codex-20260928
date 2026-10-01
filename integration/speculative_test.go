//go:build integration

package integration

import (
	"context"
	"path/filepath"
	"testing"

	"opp300-speculative-batch/internal/protocol"
)

func TestSpeculativeFailureThenSlowOriginalPublishesExactlyOnce(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, databaseURL(t))
	fixture := loadSpeculativeFixture(t)

	job := h.submit(ctx, protocol.JobSpec{
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

	finished := h.waitJobState(ctx, job.ID, protocol.JobSucceeded)
	attempts := h.attempts(ctx, job.ID)
	events := h.events(ctx, job.ID)
	merge := h.result(ctx, job.ID)

	if len(attempts) != 2 {
		t.Fatalf("attempt count = %d, want 2: %+v", len(attempts), attempts)
	}
	if attempts[0].Kind != "original" || attempts[0].State != protocol.AttemptSucceeded {
		t.Fatalf("original attempt = %+v, want succeeded original", attempts[0])
	}
	if attempts[1].Kind != "speculative" || attempts[1].State != protocol.AttemptFailed {
		t.Fatalf("speculative attempt = %+v, want failed speculative", attempts[1])
	}
	if attempts[1].FailureCategory != fixture.ExpectedFailureClass {
		t.Fatalf("speculative failure = %q, want %q", attempts[1].FailureCategory, fixture.ExpectedFailureClass)
	}

	if len(merge.Shards) != 1 {
		t.Fatalf("published shard count = %d, want 1", len(merge.Shards))
	}
	if got := merge.Shards[0].Output; got != fixture.ExpectedOutput {
		t.Fatalf("output = %q, want fixture output %q", got, fixture.ExpectedOutput)
	}
	if merge.Shards[0].AttemptID != attempts[0].ID {
		t.Fatalf("winner = %s, want slow original %s", merge.Shards[0].AttemptID, attempts[0].ID)
	}
	if finished.State != protocol.JobSucceeded || finished.Version < 3 {
		t.Fatalf("finished job = %+v, want succeeded with progressed version", finished)
	}
	assertEventType(t, events, "SPECULATION_BUDGET")
	assertEventType(t, events, "SHARD_ATTEMPT_STARTED")
	assertEventType(t, events, "ATTEMPT_FAILED")
	assertEventType(t, events, "ARTIFACT_PUBLISHED")
	assertEventType(t, events, "JOB_SUCCEEDED")
	if !eventHas(events, "ARTIFACT_PUBLISHED", "basis", "shard row lock and state transition in the publish transaction") {
		t.Fatalf("published event missing explicit decision basis: %+v", events)
	}

	stagingFiles := countFiles(t, filepath.Join(h.artifactDir, "staging"))
	if stagingFiles != 0 {
		t.Fatalf("staging files after completion = %d, want 0", stagingFiles)
	}
	publishedFiles := countFiles(t, filepath.Join(h.artifactDir, "published"))
	if publishedFiles != 1 {
		t.Fatalf("published files = %d, want 1", publishedFiles)
	}

	t.Logf("fixture=%s run=%s version=%s job=%s winner_kind=%s progress_version=%d evidence_events=%d",
		fixture.InputRunID, fixture.IDempotencyKey, fixture.Version, job.ID,
		merge.Shards[0].Kind, finished.Version, len(events))
}
