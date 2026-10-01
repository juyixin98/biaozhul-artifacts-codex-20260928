package httpapi_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"diststats/internal/config"
	"diststats/internal/httpapi"
	"diststats/internal/service"
	"diststats/internal/store"
)

type testClient struct {
	baseURL string
	t       *testing.T
	client  *http.Client
}

type apiError struct {
	Error struct {
		Kind    string `json:"kind"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func setupServer(t *testing.T) (*testClient, *store.DB) {
	t.Helper()
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		url = "postgres:///opp298?sslmode=disable"
	}
	ctx := context.Background()
	db, err := store.Open(ctx, url)
	if err != nil {
		t.Skipf("PostgreSQL is not available: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	cleanupTables(t, db)

	cfg := config.Load()
	svc := service.New(db, service.Limits{
		MaxDimensions:   cfg.MaxDimensions,
		MaxObservations: cfg.MaxObservations,
	})
	handler := httpapi.New(svc, slog.New(slog.NewTextHandler(io.Discard, nil)))
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return &testClient{baseURL: server.URL, t: t, client: server.Client()}, db
}

func cleanupTables(t *testing.T, db *store.DB) {
	t.Helper()
	ctx := context.Background()
	for _, table := range []string{"run_events", "runs", "shards", "jobs"} {
		if _, err := db.Exec(ctx, "TRUNCATE TABLE "+table+" RESTART IDENTITY CASCADE"); err != nil {
			t.Fatal(err)
		}
	}
}

func (c *testClient) request(method, path string, body any, into any) (int, http.Header) {
	c.t.Helper()
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			c.t.Fatal(err)
		}
		reader = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(context.Background(), method, c.baseURL+path, reader)
	if err != nil {
		c.t.Fatal(err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.client.Do(req)
	if err != nil {
		c.t.Fatal(err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		c.t.Fatal(err)
	}
	if into != nil && len(data) > 0 {
		if err := json.Unmarshal(data, into); err != nil {
			c.t.Fatalf("decode %s: %v: %s", path, err, string(data))
		}
	}
	return resp.StatusCode, resp.Header
}

func (c *testClient) requestRaw(method, path, body string) (int, []byte, http.Header) {
	c.t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), method, c.baseURL+path, strings.NewReader(body))
	if err != nil {
		c.t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.t.Fatal(err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		c.t.Fatal(err)
	}
	return resp.StatusCode, data, resp.Header
}

func createJob(t *testing.T, c *testClient, dim int) string {
	t.Helper()
	var job struct {
		ID string `json:"id"`
	}
	status, _ := c.request(http.MethodPost, "/v1/jobs", map[string]int{"dimension": dim}, &job)
	if status != http.StatusCreated {
		t.Fatalf("create job status = %d", status)
	}
	return job.ID
}

type shardRequest struct {
	ShardID      string           `json:"shard_id"`
	Origin       []float64        `json:"origin,omitempty"`
	Observations []map[string]any `json:"observations,omitempty"`
}

func TestDistributedWorkflowDedupeDimensionsReplayAndRunLogs(t *testing.T) {
	c, _ := setupServer(t)
	jobID := createJob(t, c, 2)

	point := func(x, y float64) map[string]any {
		return map[string]any{"values": []float64{x, y}}
	}
	first := shardRequest{
		ShardID: "north",
		Origin:  []float64{10, 20},
		Observations: []map[string]any{
			point(11, 21), point(13, 19),
		},
	}
	status, header := c.request(http.MethodPost, "/v1/jobs/"+jobID+"/shards", first, nil)
	if status != http.StatusAccepted {
		t.Fatalf("first shard status = %d", status)
	}
	if header.Get("X-Run-Number") == "" {
		t.Fatal("missing run number response header")
	}

	status, dupHeader := c.request(http.MethodPost, "/v1/jobs/"+jobID+"/shards", first, nil)
	if status != http.StatusConflict {
		t.Fatalf("duplicate shard status = %d, want 409", status)
	}
	duplicateRun := dupHeader.Get("X-Run-Number")
	var duplicateLog struct {
		Events []struct {
			Result string `json:"result"`
			Reason string `json:"reason"`
		} `json:"events"`
	}
	status, _ = c.request(http.MethodGet, "/v1/runs/"+duplicateRun, nil, &duplicateLog)
	if status != http.StatusOK || len(duplicateLog.Events) == 0 || duplicateLog.Events[0].Result != "failed" {
		t.Fatalf("duplicate rejection run log = %+v status %d", duplicateLog, status)
	}

	badDim := shardRequest{ShardID: "wrong", Observations: []map[string]any{point(1, 2)}}
	// The job has dimension 2; construct a malformed 3-dimensional observation.
	badDim.Observations = []map[string]any{{"values": []float64{1, 2, 3}}}
	var ae apiError
	status, _ = c.request(http.MethodPost, "/v1/jobs/"+jobID+"/shards", badDim, &ae)
	if status != http.StatusBadRequest || ae.Error.Kind != "INVALID_INPUT" || ae.Error.Code != "DIMENSION_MISMATCH" {
		t.Fatalf("bad dimension status=%d error=%+v", status, ae)
	}

	second := shardRequest{
		ShardID: "south",
		Origin:  []float64{10, 20},
		Observations: []map[string]any{
			point(10, 20), point(12, 22),
		},
	}
	var submit struct {
		RunNumber  int64 `json:"run_number"`
		JobSummary struct {
			N      int64     `json:"n"`
			Weight float64   `json:"weight"`
			Means  []float64 `json:"means"`
		} `json:"job_summary"`
	}
	status, _ = c.request(http.MethodPost, "/v1/jobs/"+jobID+"/shards", second, &submit)
	if status != http.StatusAccepted {
		t.Fatalf("second shard status = %d", status)
	}
	if submit.RunNumber <= 0 || submit.JobSummary.N != 4 || submit.JobSummary.Weight != 4 {
		t.Fatalf("unexpected submit response: %+v", submit)
	}

	var replay struct {
		Result struct {
			ShardCount int `json:"shard_count"`
			Metrics    struct {
				Means                []float64 `json:"means"`
				PopulationCovariance []float64 `json:"population_covariance"`
				SampleCovariance     []float64 `json:"sample_covariance"`
			} `json:"metrics"`
		} `json:"result"`
		Comparison struct {
			Accepted   bool    `json:"accepted"`
			MaxM2Error float64 `json:"max_m2_error"`
			Tolerance  float64 `json:"tolerance"`
		} `json:"comparison"`
	}
	status, replayHeader := c.request(http.MethodGet,
		"/v1/jobs/"+jobID+"/replay?order=balanced&compare=true", nil, &replay)
	if status != http.StatusOK {
		t.Fatalf("replay status = %d", status)
	}
	if replay.Result.ShardCount != 2 || !replay.Comparison.Accepted {
		t.Fatalf("replay mismatch: %+v", replay)
	}
	assertApprox(t, replay.Result.Metrics.Means[0], 11.5, 1e-10)
	assertApprox(t, replay.Result.Metrics.Means[1], 20.5, 1e-10)
	assertApprox(t, replay.Result.Metrics.PopulationCovariance[0], 1.25, 1e-10)
	assertApprox(t, replay.Result.Metrics.SampleCovariance[0], 5.0/3.0, 1e-10)
	t.Logf("balanced replay measured m2 error %.3e tolerance %.3e",
		replay.Comparison.MaxM2Error, replay.Comparison.Tolerance)

	var run struct {
		Number int64 `json:"number"`
		Events []struct {
			Stage  string `json:"stage"`
			Result string `json:"result"`
			Reason string `json:"reason"`
			State  map[string]any
		} `json:"events"`
	}
	runNumber := replayHeader.Get("X-Run-Number")
	status, _ = c.request(http.MethodGet, "/v1/runs/"+runNumber, nil, &run)
	if status != http.StatusOK || len(run.Events) < 2 {
		t.Fatalf("run log status=%d events=%d", status, len(run.Events))
	}
	var sawCheckpoint bool
	for _, e := range run.Events {
		if strings.Contains(e.Reason, "merge tree") {
			sawCheckpoint = true
		}
	}
	if !sawCheckpoint {
		t.Fatalf("run log lacks replay judgement reason: %+v", run.Events)
	}

	status, _ = c.request(http.MethodPost, "/v1/jobs/"+jobID+"/finalize", nil, nil)
	if status != http.StatusOK {
		t.Fatalf("finalize status = %d", status)
	}
	status, _ = c.request(http.MethodPost, "/v1/jobs/"+jobID+"/shards", second, nil)
	if status != http.StatusConflict {
		t.Fatalf("post-finalize shard status = %d, want 409", status)
	}
}

func TestErrorCategoriesAreSpecific(t *testing.T) {
	c, _ := setupServer(t)
	var ae apiError

	status, body, _ := c.requestRaw(http.MethodPost, "/v1/jobs", `{"dimension":-1}`)
	_ = json.Unmarshal(body, &ae)
	if status != http.StatusBadRequest || ae.Error.Code != "INVALID_DIMENSION" {
		t.Fatalf("invalid dimension status=%d body=%s", status, body)
	}

	status, body, _ = c.requestRaw(http.MethodPost, "/v1/jobs", `{"dimension":999999}`)
	_ = json.Unmarshal(body, &ae)
	if status != http.StatusUnprocessableEntity || ae.Error.Kind != "RESOURCE_EXHAUSTED" {
		t.Fatalf("dimension limit status=%d body=%s", status, body)
	}

	status, body, _ = c.requestRaw(http.MethodPost, "/v1/jobs", `{"dimension":`)
	if status != http.StatusBadRequest || !strings.Contains(string(body), "MALFORMED_JSON") {
		t.Fatalf("malformed JSON status=%d body=%s", status, body)
	}

	ae = apiError{}
	status, body, _ = c.requestRaw(http.MethodGet, "/v1/jobs/"+zeroUUID()+"/result", "")
	_ = json.Unmarshal(body, &ae)
	if status != http.StatusNotFound || ae.Error.Kind != "NOT_FOUND" {
		t.Fatalf("missing job status=%d body=%s", status, body)
	}

	status, body, _ = c.requestRaw(http.MethodPost,
		"/v1/jobs/"+createJob(t, c, 1)+"/shards",
		`{"shard_id":"x","observations":[{"values":[1],"weight":1e308},{"values":[2],"weight":1e308}]}`)
	ae = apiError{}
	_ = json.Unmarshal(body, &ae)
	_ = math.NaN
	if status != http.StatusUnprocessableEntity || ae.Error.Kind != "COMPUTATION_FAILED" {
		t.Fatalf("computation overflow status=%d body=%s", status, body)
	}
}

func zeroUUID() string {
	return "00000000-0000-0000-0000-000000000000"
}

func assertApprox(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("got %.17g want %.17g", got, want)
	}
}

func TestHealth(t *testing.T) {
	c, _ := setupServer(t)
	status, _ := c.request(http.MethodGet, "/healthz", nil, nil)
	if status != http.StatusOK {
		t.Fatalf("health status = %d", status)
	}
}

var _ = fmt.Sprintf
