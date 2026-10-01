package store

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"specbatch/internal/kernel"
	"specbatch/internal/model"
)

func openTestStore(t *testing.T) (*Store, context.Context) {
	t.Helper()
	base := testDatabaseURL()
	ctx := context.Background()
	admin, err := Open(ctx, base)
	if err != nil {
		t.Skipf("PostgreSQL unavailable: %v", err)
	}
	schema := fmt.Sprintf("it_%d", time.Now().UnixNano())
	if _, err := admin.DB().ExecContext(ctx, `DROP SCHEMA IF EXISTS `+schema+` CASCADE; CREATE SCHEMA `+schema); err != nil {
		t.Skipf("cannot create isolated schema: %v", err)
	}
	admin.Close()
	dsn := withSchema(base, schema)
	st, err := Open(ctx, dsn)
	if err != nil {
		t.Skipf("cannot open isolated schema: %v", err)
	}
	t.Cleanup(func() {
		_ = st.Close()
	})
	return st, ctx
}

func testManifest(jobID string, attempts int, specCopies int) model.Manifest {
	return model.Manifest{
		SchemaVersion: "specbatch/manifest/v1", JobID: jobID, Kernel: kernel.Name,
		KernelVersion: kernel.Version, ProtocolVersion: "specbatch/v1",
		MaxAttemptsPerShard: attempts, MaxSpeculativeCopies: specCopies, SpecAfterMillis: 20,
		Shards: []model.ManifestShard{{
			ID: "s1", Records: []model.InputRecord{{ID: "r1", Value: 3}},
			Behavior: map[string]int{"1_delay_ms": 1000, "2_delay_ms": 1},
		}},
	}
}

func artifactFor(lease model.Lease) (model.ArtifactDocument, []byte) {
	doc, err := kernel.CanonicalArtifact(lease)
	if err != nil {
		panic(err)
	}
	raw, _ := json.Marshal(doc)
	return doc, raw
}

func TestConcurrentSuccessArbitratesExactlyOneWinner(t *testing.T) {
	st, ctx := openTestStore(t)
	if _, err := st.Bootstrap(ctx); err != nil {
		t.Fatal(err)
	}
	if err := st.CreateJob(ctx, testManifest("concurrent-winner", 2, 1)); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Schedule(ctx, 1, time.Now()); err != nil {
		t.Fatal(err)
	}
	claim1, ok, err := st.Claim(ctx, "w1", 1)
	if err != nil || !ok {
		t.Fatalf("initial claim ok=%v err=%v", ok, err)
	}
	time.Sleep(30 * time.Millisecond)
	if _, err := st.Schedule(ctx, 1, time.Now()); err != nil {
		t.Fatal(err)
	}
	claim2, ok, err := st.Claim(ctx, "w2", 1)
	if err != nil || !ok {
		t.Fatalf("speculative claim ok=%v err=%v", ok, err)
	}
	t.Logf("claims: first no=%d worker lease=%q second no=%d lease worker=%q", claim1.Lease.AttemptNo, claim1.Lease.Token, claim2.Lease.AttemptNo, claim2.Lease.Token)
	attemptsBefore, err := st.Attempts(ctx, "concurrent-winner")
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range attemptsBefore {
		t.Logf("pre-submit attempt %d worker=%q status=%s category=%s", a.AttemptNo, a.WorkerID, a.Status, a.FailureCategory)
	}

	results := make([]SubmitResult, 2)
	errs := make([]error, 2)
	start := make(chan struct{})
	var wg sync.WaitGroup
	submit := func(i int, claim Claim) {
		defer wg.Done()
		<-start
		doc, raw := artifactFor(claim.Lease)
		r, err := st.SubmitSuccess(ctx, SubmitInput{
			Token: claim.Lease.Token, WorkerID: "w" + fmt.Sprint(i+1),
			Document: doc, Content: raw,
		})
		results[i], errs[i] = r, err
	}
	wg.Add(2)
	go submit(0, claim1)
	go submit(1, claim2)
	close(start)
	wg.Wait()

	winners := 0
	late := 0
	for i, err := range errs {
		if err != nil {
			if !errorIs(err, ErrAttemptExpired) {
				t.Errorf("submission %d returned unexpected error: %v", i, err)
			}
		}
		if err == nil && results[i].Won {
			winners++
		} else if err == nil && !results[i].Won {
			late++
		} else {
			late++
		}
	}
	if winners != 1 || late != 1 {
		t.Fatalf("winners=%d late=%d, want exactly one winner", winners, late)
	}
	status, err := st.JobStatus(ctx, "concurrent-winner", true)
	if err != nil {
		t.Fatalf("job status: %#v", err)
	}
	if status.Job.Status != model.JobSucceeded || len(status.ShardStatuses) != 1 || status.ShardStatuses[0].Status != model.ShardSucceeded {
		t.Fatalf("job/shard not succeeded: %+v", status)
	}
	if len(status.WinnerByShard) != 1 || status.WinnerByShard["s1"] == 0 {
		t.Fatalf("missing unique winner map: %+v", status.WinnerByShard)
	}
	categories := map[string]int{}
	for _, a := range status.Attempts {
		categories[a.Status+":"+a.FailureCategory]++
	}
	if categories["succeeded:"] != 1 || categories["rejected:"+model.FailureSupersededSubmission] != 1 {
		t.Fatalf("unexpected terminal attempt categories: %#v", categories)
	}
	merge, err := st.Merged(ctx, "concurrent-winner")
	if err != nil {
		t.Fatal(err)
	}
	if merge.RecordCount != 1 || merge.Records[0].ID != "r1" || merge.Records[0].Value != 6 {
		t.Fatalf("unexpected canonical merge: %+v", merge)
	}
}

func TestBootstrapFencesPreviousEpochAndRetryCanSucceed(t *testing.T) {
	st, ctx := openTestStore(t)
	epoch1, err := st.Bootstrap(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.CreateJob(ctx, testManifest("restart-fence", 3, 1)); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Schedule(ctx, epoch1, time.Now()); err != nil {
		t.Fatal(err)
	}
	old, ok, err := st.Claim(ctx, "old-worker", epoch1)
	if err != nil || !ok {
		t.Fatalf("claim old ok=%v err=%v", ok, err)
	}
	epoch2, err := st.Bootstrap(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if epoch2 != epoch1+1 {
		t.Fatalf("epoch=%d want %d", epoch2, epoch1+1)
	}
	if _, err := st.Heartbeat(ctx, old.Lease.Token, "old-worker", .5); !errorIs(err, ErrAttemptExpired) {
		t.Fatalf("old heartbeat err=%v want expired", err)
	}
	oldDoc, oldRaw := artifactFor(old.Lease)
	_, err = st.SubmitSuccess(ctx, SubmitInput{Token: old.Lease.Token, WorkerID: "old-worker", Document: oldDoc, Content: oldRaw})
	if !errorIs(err, ErrAttemptExpired) {
		t.Fatalf("old submit err=%v want expired/fenced", err)
	}
	if _, err := st.Schedule(ctx, epoch2, time.Now()); err != nil {
		t.Fatal(err)
	}
	next, ok, err := st.Claim(ctx, "new-worker", epoch2)
	if err != nil || !ok {
		t.Fatalf("new claim ok=%v err=%v", ok, err)
	}
	if next.Lease.AttemptNo != 2 || next.Lease.Kind != model.AttemptRetry {
		t.Fatalf("unexpected new lease: %+v", next.Lease)
	}
	doc, raw := artifactFor(next.Lease)
	result, err := st.SubmitSuccess(ctx, SubmitInput{Token: next.Lease.Token, WorkerID: "new-worker", Document: doc, Content: raw})
	if err != nil || !result.Won {
		t.Fatalf("new submit result=%+v err=%v", result, err)
	}
	status, _ := st.JobStatus(ctx, "restart-fence", true)
	foundRestartFailure := false
	for _, a := range status.Attempts {
		if a.AttemptNo == 1 && a.Status == model.AttemptFailed && a.FailureCategory == model.FailureCoordinatorRestart {
			foundRestartFailure = true
		}
	}
	if !foundRestartFailure {
		t.Fatalf("old attempt not marked with restart failure: %+v", status.Attempts)
	}
}

func errorIs(err, target error) bool {
	return err != nil && (err == target || containsText(err.Error(), target.Error()))
}

func containsText(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
