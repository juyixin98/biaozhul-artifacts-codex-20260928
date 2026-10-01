package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"diststats/internal/httpapi"
	"diststats/internal/postgres"
	"diststats/internal/protocol"
	"diststats/internal/service"
)

type replayResponse struct {
	Result protocol.JobResult `json:"result"`
	Events []protocol.Event   `json:"events"`
}

func TestHTTPPostgresAggregationDedupDimensionAndReplay(t *testing.T) {
	ctx := context.Background()
	pool := openTestPool(t, ctx)
	_, err := pool.Exec(ctx, "TRUNCATE diststats_b.jobs CASCADE")
	if err != nil {
		t.Fatalf("truncate test schema: %v", err)
	}

	var logs bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&logs, nil))
	store := postgres.NewStore(pool)
	server := httptest.NewServer(httpapi.New(service.New(store), logger).Handler())
	defer server.Close()

	runID := "run-integration-" + time.Now().Format("20060102T150405.000000000")
	job := createJob(t, server.URL, protocol.JobRequest{
		RunID:          runID,
		Dimension:      2,
		Estimator:      protocol.Population,
		ExpectedShards: 2,
	})

	// Submit the one-row shard first, then the two-row shard. This exercises
	// the unstable partition ordering explicitly.
	submitShard(t, server.URL, job.Job.ID, protocol.ShardRequest{
		ShardID: "shard-b",
		Values:  []float64{4, 8},
	}, http.StatusAccepted)
	submitShard(t, server.URL, job.Job.ID, protocol.ShardRequest{
		ShardID: "shard-a",
		Values:  []float64{0, 0, 2, 4},
	}, http.StatusAccepted)

	result := getJob(t, server.URL, job.Job.ID)
	if result.Job.Status != "complete" || result.Job.ReceivedShards != 2 {
		t.Fatalf("job = status %q received %d, want complete/2", result.Job.Status, result.Job.ReceivedShards)
	}
	assertFloat(t, result.Summary.Weight, 3, 0)
	assertFloat(t, result.Summary.Mean[0], 2, 0)
	assertFloat(t, result.Summary.Mean[1], 4, 0)
	assertFloat(t, *result.Summary.Variance[0], 8.0/3.0, 1e-12)
	assertFloat(t, *result.Summary.Variance[1], 32.0/3.0, 1e-12)
	assertFloat(t, *result.Summary.Covariance[1], 16.0/3.0, 1e-12)

	submitShard(t, server.URL, job.Job.ID, protocol.ShardRequest{
		ShardID: "shard-a",
		Values:  []float64{9, 9},
	}, http.StatusConflict)
	submitShard(t, server.URL, job.Job.ID, protocol.ShardRequest{
		ShardID: "shard-c",
		Values:  []float64{1},
	}, http.StatusBadRequest)

	replay := getReplay(t, server.URL, job.Job.ID)
	if len(replay.Events) != 4 {
		t.Fatalf("replay event count = %d, want 4", len(replay.Events))
	}
	for i, event := range replay.Events {
		if event.Sequence != int64(i+1) {
			t.Fatalf("event %d has sequence %d", i, event.Sequence)
		}
		if event.RunID != runID {
			t.Fatalf("event %d missing run id, got %q", i, event.RunID)
		}
	}
	if replay.Events[0].Type != "shard_accepted" || replay.Events[0].ShardID != "shard-b" ||
		replay.Events[0].After == nil || replay.Events[0].After.Count != 1 {
		t.Fatalf("first replay event missing key post-state: %+v", replay.Events[0])
	}
	if replay.Events[2].ErrorClass != string(protocol.ClassState) ||
		replay.Events[2].Reason != "DUPLICATE_SHARD" {
		t.Fatalf("duplicate event mismatch: %+v", replay.Events[2])
	}
	if replay.Events[3].ErrorClass != string(protocol.ClassInput) ||
		replay.Events[3].Reason != "INVALID_VALUES_SHAPE" {
		t.Fatalf("dimension/shape event mismatch: %+v", replay.Events[3])
	}

	if !strings.Contains(logs.String(), runID) || !strings.Contains(logs.String(), "DUPLICATE_SHARD") {
		t.Fatalf("structured logs did not retain run id and failure reason:\n%s", logs.String())
	}
}

func TestHTTPFailureClassesAreDistinguishable(t *testing.T) {
	ctx := context.Background()
	pool := openTestPool(t, ctx)
	server := httptest.NewServer(httpapi.New(service.New(postgres.NewStore(pool)), slog.New(slog.NewTextHandler(io.Discard, nil))).Handler())
	defer server.Close()

	job := createJob(t, server.URL, protocol.JobRequest{
		RunID:          "run-failures-" + time.Now().Format("150405.000000000"),
		Dimension:      1,
		Estimator:      protocol.Population,
		ExpectedShards: 2,
	})

	badBody := append(bytes.Repeat([]byte(" "), 5<<20), []byte(`{"shard_id":"large"}`)...)
	response, err := http.Post(server.URL+"/v1/jobs/"+job.Job.ID+"/shards", "application/json", bytes.NewReader(badBody))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("large body status = %d, want 413", response.StatusCode)
	}
	var failure protocol.ErrorResponse
	decodeBody(t, response.Body, &failure)
	if failure.ErrorClass != string(protocol.ClassResourceExhausted) || failure.Code != "REQUEST_TOO_LARGE" {
		t.Fatalf("large body error = %+v", failure)
	}

	status, failure := submitShardExpectingError(t, server.URL, job.Job.ID, protocol.ShardRequest{
		ShardID: "overflow",
		Values:  []float64{1, 2},
		Weights: []float64{math.MaxFloat64, math.MaxFloat64},
	})
	if status != http.StatusUnprocessableEntity ||
		failure.ErrorClass != string(protocol.ClassComputation) || failure.Code != "WEIGHT_OVERFLOW" {
		t.Fatalf("overflow status=%d error=%+v", status, failure)
	}
}

func openTestPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("TEST_DATABASE_URL")
	if databaseURL == "" {
		databaseURL = "postgres:///opp298_test?sslmode=disable"
	}
	pool, err := postgres.Connect(ctx, databaseURL, 4)
	if err != nil {
		t.Fatalf("connect test database: %v", err)
	}
	t.Cleanup(pool.Close)
	if err := postgres.Migrate(ctx, pool, "../../migrations"); err != nil {
		t.Fatalf("migrate test database: %v", err)
	}
	return pool
}

func createJob(t *testing.T, baseURL string, request protocol.JobRequest) protocol.JobResult {
	t.Helper()
	response, err := http.Post(baseURL+"/v1/jobs", "application/json", jsonReader(request))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		t.Fatalf("create job status=%d body=%s", response.StatusCode, readAll(t, response.Body))
	}
	var result protocol.JobResult
	decodeBody(t, response.Body, &result)
	return result
}

func submitShard(t *testing.T, baseURL, jobID string, shard protocol.ShardRequest, wantStatus int) {
	t.Helper()
	status, _ := submitShardExpectingError(t, baseURL, jobID, shard)
	if status != wantStatus {
		t.Fatalf("shard %q status=%d, want %d", shard.ShardID, status, wantStatus)
	}
}

func submitShardExpectingError(t *testing.T, baseURL, jobID string, shard protocol.ShardRequest) (int, protocol.ErrorResponse) {
	t.Helper()
	response, err := http.Post(baseURL+"/v1/jobs/"+jobID+"/shards", "application/json", jsonReader(shard))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusAccepted {
		var result protocol.ShardResult
		decodeBody(t, response.Body, &result)
		return response.StatusCode, protocol.ErrorResponse{}
	}
	var failure protocol.ErrorResponse
	decodeBody(t, response.Body, &failure)
	return response.StatusCode, failure
}

func getJob(t *testing.T, baseURL, jobID string) protocol.JobResult {
	t.Helper()
	response, err := http.Get(baseURL + "/v1/jobs/" + jobID)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("get job status=%d body=%s", response.StatusCode, readAll(t, response.Body))
	}
	var result protocol.JobResult
	decodeBody(t, response.Body, &result)
	return result
}

func getReplay(t *testing.T, baseURL, jobID string) replayResponse {
	t.Helper()
	response, err := http.Get(baseURL + "/v1/jobs/" + jobID + "/replay")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("replay status=%d body=%s", response.StatusCode, readAll(t, response.Body))
	}
	var result replayResponse
	decodeBody(t, response.Body, &result)
	return result
}

func jsonReader(value any) io.Reader {
	data, _ := json.Marshal(value)
	return bytes.NewReader(data)
}

func decodeBody(t *testing.T, body io.Reader, target any) {
	t.Helper()
	if err := json.NewDecoder(body).Decode(target); err != nil {
		t.Fatalf("decode response: %v", err)
	}
}

func readAll(t *testing.T, body io.Reader) string {
	t.Helper()
	data, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func assertFloat(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("float = %.17g, want %.17g (tol %.3e)", got, want, tolerance)
	}
}
