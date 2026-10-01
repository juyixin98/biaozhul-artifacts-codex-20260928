package integration

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"specbatch/internal/model"
)

func TestNormalRunSpeculativeArbitrationAndCleanup(t *testing.T) {
	e := setupEnv(t)
	port := freePort(t)
	base := "http://127.0.0.1:" + itoa(port)
	e.startCoordinator(t, port)
	workers := []*process{
		e.startWorker(t, port, "normal-w1"),
		e.startWorker(t, port, "normal-w2"),
		e.startWorker(t, port, "normal-w3"),
		e.startWorker(t, port, "normal-w4"),
	}
	_ = workers

	jobID := "case-normal-" + uniqueSuffix()
	manifest := loadFixture(t, "testdata/fixtures/normal_job.json", jobID)
	postJSON(t, base, "/v1/jobs", map[string]any{"manifest": manifest})
	status := waitJob(t, base, jobID, 8*time.Second)

	if status.Job.Status != model.JobSucceeded {
		t.Fatalf("status=%s, want succeeded; attempts=%+v", status.Job.Status, status.Attempts)
	}
	if len(status.ShardStatuses) != 2 {
		t.Fatalf("shard count=%d, want 2", len(status.ShardStatuses))
	}

	var merge model.MergeDocument
	getJSON(t, base, "/v1/jobs/"+jobID+"/merge?persist=1", &merge)
	expected := loadExpected(t, "testdata/expected/normal_expected.json", jobID)
	if merge.RecordCount != 3 {
		t.Fatalf("merged count=%d, want 3", merge.RecordCount)
	}
	if !recordsEqual(merge.Records, expected.Records) {
		t.Fatalf("merge mismatch:\n got=%+v\nwant=%+v", merge.Records, expected.Records)
	}
	seenIDs := map[string]int{}
	for _, rec := range merge.Records {
		seenIDs[rec.ID]++
	}
	for id, count := range seenIDs {
		if count != 1 {
			t.Fatalf("record %s appeared %d times in merge", id, count)
		}
	}

	byShard := groupAttempts(status.Attempts)
	alpha := byShard["shard-alpha"]
	if len(alpha) != 2 {
		t.Fatalf("alpha attempts=%d, want initial + speculative", len(alpha))
	}
	assertTerminalCounts(t, alpha, map[string]map[string]int{
		"succeeded": {"": 1},
		"rejected":  {model.FailureSupersededSubmission: 1},
	})
	if winner := status.WinnerByShard["shard-alpha"]; winner != 2 {
		t.Fatalf("alpha winner=%d, want fast speculative attempt 2", winner)
	}

	beta := byShard["shard-beta"]
	if len(beta) != 2 {
		t.Fatalf("beta attempts=%d, want initial + failed speculative", len(beta))
	}
	assertTerminalCounts(t, beta, map[string]map[string]int{
		"succeeded": {"": 1},
		"failed":    {model.FailureSimulatedWorker: 1},
	})
	if winner := status.WinnerByShard["shard-beta"]; winner != 1 {
		t.Fatalf("beta winner=%d, want slow initial attempt 1", winner)
	}

	mergedPath := filepath.Join(e.root, "artifacts", jobID, "merged.json")
	if _, err := os.Stat(mergedPath); err != nil {
		t.Fatalf("published merged file missing: %v", err)
	}
	for _, shardID := range []string{"shard-alpha", "shard-beta"} {
		p := filepath.Join(e.root, "artifacts", jobID, "shards", shardID+".json")
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("published shard %s missing: %v", shardID, err)
		}
	}
	assertNoFileUnder(t, filepath.Join(e.root, "artifacts"), func(p string) bool {
		name := filepath.Base(p)
		return strings.HasPrefix(name, ".upload-")
	})
	assertNoFileUnder(t, filepath.Join(e.root, "staging"), func(p string) bool {
		return filepath.Base(p) == "artifact.json"
	})

	coordLog := readFile(t, filepath.Join(e.root, "coordinator-"+itoa(port)+".log"))
	assertContainsAll(t, coordLog,
		`"event":"coordinator_ready"`, `"kernel_version":"v1.0.0"`, `"kind":"speculative"`,
		`"event":"scheduler_launch"`, jobID,
	)
	for _, id := range []string{"normal-w1", "normal-w2", "normal-w3", "normal-w4"} {
		logText := readFile(t, filepath.Join(e.root, "worker-"+id+".log"))
		if strings.Contains(logText, jobID) {
			assertContainsAll(t, logText, `"run_id":"`, `"event":"attempt_started"`, `"kernel_version":"v1.0.0"`,
				`"record_ids"`, `"event":"heartbeat"`)
		}
	}
}

func TestCoordinatorRestartFencesOldEpochAndCompletesJob(t *testing.T) {
	e := setupEnv(t)
	port := freePort(t)
	base := "http://127.0.0.1:" + itoa(port)
	oldCoord := e.startCoordinator(t, port)
	e.startWorker(t, port, "restart-old-worker")
	jobID := "case-restart-" + uniqueSuffix()
	manifest := loadFixture(t, "testdata/fixtures/restart_job.json", jobID)
	postJSON(t, base, "/v1/jobs", map[string]any{"manifest": manifest})

	// Wait until the old worker owns the slow initial attempt.
	deadline := time.Now().Add(3 * time.Second)
	var oldClaim bool
	for time.Now().Before(deadline) && !oldClaim {
		logText := readFile(t, filepath.Join(e.root, "worker-restart-old-worker.log"))
		oldClaim = strings.Contains(logText, jobID) && strings.Contains(logText, `"attempt_started"`)
		time.Sleep(20 * time.Millisecond)
	}
	if !oldClaim {
		t.Fatal("old worker never claimed restart fixture")
	}
	oldCoord.KillForRestart(t)
	time.Sleep(250 * time.Millisecond)

	newCoord := e.startCoordinator(t, port)
	_ = newCoord
	e.startWorker(t, port, "restart-new-worker")
	status := waitJob(t, base, jobID, 8*time.Second)
	if status.Job.Status != model.JobSucceeded {
		t.Fatalf("status=%s, want succeeded; attempts=%+v", status.Job.Status, status.Attempts)
	}
	if status.WinnerByShard["shard-restart"] != 2 {
		t.Fatalf("winner=%v, want post-restart retry attempt 2", status.WinnerByShard)
	}
	counts := map[string]int{}
	for _, a := range status.Attempts {
		counts[a.Status+":"+a.FailureCategory]++
	}
	if counts["failed:"+model.FailureCoordinatorRestart] != 1 {
		t.Fatalf("missing restart-fenced initial attempt: %#v", counts)
	}
	if counts["succeeded:"] != 1 {
		t.Fatalf("missing one successful retry: %#v", counts)
	}

	var merge model.MergeDocument
	getJSON(t, base, "/v1/jobs/"+jobID+"/merge?persist=1", &merge)
	expected := loadExpected(t, "testdata/expected/restart_expected.json", jobID)
	if !recordsEqual(merge.Records, expected.Records) || merge.RecordCount != 1 {
		t.Fatalf("restart merge mismatch: got=%+v want=%+v", merge, expected)
	}

	coordLog := readFile(t, filepath.Join(e.root, "coordinator-"+itoa(port)+".log"))
	if strings.Count(coordLog, `"event":"coordinator_ready"`) < 2 {
		t.Fatalf("expected two coordinator generations in log, got: %s", coordLog)
	}
	if !strings.Contains(coordLog, model.FailureCoordinatorRestart) {
		t.Fatal("coordinator log lacks restart failure category")
	}
	oldWorkerLog := readFile(t, filepath.Join(e.root, "worker-restart-old-worker.log"))
	if !strings.Contains(oldWorkerLog, "heartbeat_rejected") && !strings.Contains(oldWorkerLog, "late_submission_rejected_or_transport_failed") {
		t.Fatal("old worker log does not show fenced heartbeat or late submission")
	}
	assertNoFileUnder(t, filepath.Join(e.root, "staging"), func(p string) bool {
		return filepath.Base(p) == "artifact.json"
	})
}

func TestUnknownJobIsNotMappedToSuccess(t *testing.T) {
	e := setupEnv(t)
	port := freePort(t)
	e.startCoordinator(t, port)
	resp, err := http.Get("http://127.0.0.1:" + itoa(port) + "/v1/jobs/does-not-exist")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status=%d, want 404", resp.StatusCode)
	}
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["failure_category"] != "job_not_found" {
		t.Fatalf("unknown job body=%v", body)
	}
}

func groupAttempts(attempts []model.Attempt) map[string][]model.Attempt {
	out := map[string][]model.Attempt{}
	for _, a := range attempts {
		out[a.ShardID] = append(out[a.ShardID], a)
	}
	return out
}

func assertTerminalCounts(t *testing.T, attempts []model.Attempt, want map[string]map[string]int) {
	t.Helper()
	got := map[string]map[string]int{}
	for _, a := range attempts {
		if got[a.Status] == nil {
			got[a.Status] = map[string]int{}
		}
		got[a.Status][a.FailureCategory]++
	}
	for status, cats := range want {
		for cat, count := range cats {
			if got[status][cat] != count {
				t.Fatalf("terminal counts mismatch for %s/%s: got=%#v want=%#v; attempts=%+v", status, cat, got, want, attempts)
			}
		}
	}
}

func recordsEqual(a, b []model.OutputRecord) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
