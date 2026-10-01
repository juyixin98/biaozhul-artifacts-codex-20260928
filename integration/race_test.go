//go:build integration

package integration

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"path/filepath"
	"sync"
	"testing"

	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/protocol"
)

func TestConcurrentCompleteRequestsPublishOnlyOneWinner(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, databaseURL(t))
	fixture := loadRaceFixture(t)

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

	attempts := h.attempts(ctx, job.ID)
	waitForAttemptCount(t, ctx, h, job.ID, 2)
	attempts = h.attempts(ctx, job.ID)
	if len(attempts) != 2 {
		t.Fatalf("attempt count after both start = %d, want parallel original + speculative", len(attempts))
	}

	payload := []byte(fixture.ExpectedOutput)
	sum := sha256.Sum256(payload)
	checksum := hex.EncodeToString(sum[:])

	var wg sync.WaitGroup
	statuses := make(chan int, len(attempts))
	bodies := make(chan string, len(attempts))
	for _, attempt := range attempts {
		wg.Add(1)
		go func(attempt protocol.Attempt) {
			defer wg.Done()
			staged, err := stageArtifact(ctx, h.coordinatorURL, attempt.ID, h.epoch, checksum, payload)
			if err != nil {
				t.Errorf("stage for %s: %v", attempt.ID, err)
				return
			}
			status, body := completeAttempt(ctx, t, h.coordinatorURL, attempt.ID, participant.CompleteRequest{
				Epoch:          h.epoch,
				StagingID:      staged.StagingID,
				ChecksumSHA256: checksum,
				SizeBytes:      int64(len(payload)),
			})
			statuses <- status
			bodies <- body
		}(attempt)
	}
	wg.Wait()
	close(statuses)
	close(bodies)

	var success, conflict int
	for status := range statuses {
		switch status {
		case 200:
			success++
		case 409:
			conflict++
		default:
			t.Fatalf("unexpected complete status %d body=%s", status, <-bodies)
		}
	}
	if success != 1 || conflict != 1 {
		t.Fatalf("complete results success=%d conflict=%d, want exactly one each", success, conflict)
	}

	updated := h.attempts(ctx, job.ID)
	states := map[string]int{}
	for _, attempt := range updated {
		states[attempt.State]++
	}
	if states[protocol.AttemptSucceeded] != 1 || states[protocol.AttemptSuperseded] != 1 {
		t.Fatalf("attempt states = %+v, want one succeeded and one superseded", states)
	}
	merge := h.result(ctx, job.ID)
	if len(merge.Shards) != 1 {
		t.Fatalf("merge shard count = %d, want 1", len(merge.Shards))
	}
	if merge.Shards[0].Output != fixture.ExpectedOutput {
		t.Fatalf("merge output = %q, want %q", merge.Shards[0].Output, fixture.ExpectedOutput)
	}

	if stagingFiles := countFiles(t, filepath.Join(h.artifactDir, "staging")); stagingFiles != 0 {
		t.Fatalf("staging files = %d, want 0", stagingFiles)
	}
	if publishedFiles := countFiles(t, filepath.Join(h.artifactDir, "published")); publishedFiles != 1 {
		t.Fatalf("published files = %d, want one immutable artifact", publishedFiles)
	}
	t.Logf("fixture=%s run=%s version=%s job=%s concurrent_attempts=%d winner=%s",
		fixture.InputRunID, fixture.IDempotencyKey, fixture.Version, job.ID,
		len(attempts), merge.Shards[0].AttemptID)
}
