//go:build integration

package service_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"opp297/topk/internal/domain"
	"opp297/topk/internal/fixture"
	"opp297/topk/internal/participant"
	"opp297/topk/internal/service"
	"opp297/topk/internal/storage"
)

const integrationSecret = "integration-secret-32-bytes-ok"

type topKPayload struct {
	Generation string `json:"generation"`
	K          int    `json:"k"`
}

type topKResponse struct {
	RequestID string `json:"request_id"`
	Exact     bool   `json:"exact"`
	Complete  bool   `json:"complete"`
	Results   []struct {
		ID    domain.ItemID `json:"id"`
		Score int64         `json:"score"`
	} `json:"results"`
	Proof struct {
		Rounds        int    `json:"rounds"`
		TotalRowsRead int    `json:"total_rows_read"`
		Reason        string `json:"reason"`
	} `json:"stop_proof"`
	Failure *struct {
		Class   string `json:"class"`
		Message string `json:"message"`
	} `json:"failure"`
	Shards []struct {
		ID         int         `json:"id"`
		Exhausted  bool        `json:"exhausted"`
		Active     bool        `json:"active"`
		UpperBound *domain.Key `json:"upper_bound"`
		RowsRead   int         `json:"rows_read"`
	} `json:"shard_statuses"`
}

type replayResponse struct {
	Query struct {
		RequestID string `json:"request_id"`
		Exact     bool   `json:"exact"`
		Complete  bool   `json:"complete"`
	} `json:"query"`
	Events []struct {
		Kind         string      `json:"kind"`
		ShardID      int         `json:"shard_id"`
		Reason       string      `json:"reason"`
		FailureClass string      `json:"failure_class"`
		UpperBound   *domain.Key `json:"upper_bound_key"`
	} `json:"events"`
}

func TestHTTPPostgresTopKReplayAndFaults(t *testing.T) {
	ctx := context.Background()
	dbURL := integrationDatabaseURL()
	gen := domain.GenerationID(fmt.Sprintf("%s-it-%d", fixture.DefaultGeneration, time.Now().UnixNano()))
	store, closeStore, err := storage.Open(ctx, dbURL)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer closeStore()
	if err := store.EnsureSchema(ctx); err != nil {
		t.Fatalf("schema: %v", err)
	}
	records := fixture.Default()
	for i := range records {
		records[i].Generation = gen
	}
	if err := store.SeedGeneration(ctx, gen, records); err != nil {
		t.Fatalf("seed: %v", err)
	}
	t.Cleanup(func() { cleanupGeneration(context.Background(), t, dbURL, gen) })

	shards := make([]*httptest.Server, 4)
	urls := map[int]string{}
	for id := range shards {
		id := id
		srv := httptest.NewServer((&participant.Server{
			Store: store, ShardID: id, BoundSecret: []byte(integrationSecret),
			AllowFault: true, Logger: testLogger(t),
		}).Routes())
		shards[id] = srv
		urls[id] = srv.URL + "/v1/shards/page"
		t.Cleanup(srv.Close)
	}

	coordinator := &service.Coordinator{
		Store: store, ShardIDs: []int{0, 1, 2, 3}, ShardURLs: urls,
		PageSize: 3, Timeout: 100 * time.Millisecond,
		BoundSecret: []byte(integrationSecret), Logger: testLogger(t),
	}
	api := httptest.NewServer(coordinator.Routes())
	defer api.Close()

	t.Run("exact_results_compare_independent_fixture_and_replay", func(t *testing.T) {
		resp := postTopK(t, api.URL, topKPayload{Generation: string(gen), K: 6}, "")
		if !resp.Exact || !resp.Complete {
			t.Fatalf("exact=%v complete=%v failure=%v", resp.Exact, resp.Complete, resp.Failure)
		}
		want := fixture.ExpectedTopK(6)
		if len(resp.Results) != len(want) {
			t.Fatalf("len(results)=%d want=%d", len(resp.Results), len(want))
		}
		for i, got := range resp.Results {
			if got.ID != want[i].ID || got.Score != want[i].Score {
				t.Fatalf("position %d got=(%s,%d) want=%v", i+1, got.ID, got.Score, want[i])
			}
		}
		if resp.Proof.TotalRowsRead != 15 {
			t.Fatalf("rows=%d want=15 proof=%+v", resp.Proof.TotalRowsRead, resp.Proof)
		}
		if resp.Proof.Reason == "" || resp.Proof.Rounds < 2 {
			t.Fatalf("missing stop proof: %+v", resp.Proof)
		}
		var emptyFound bool
		for _, st := range resp.Shards {
			if st.ID == 3 {
				emptyFound = true
				if !st.Exhausted || st.Active || st.RowsRead != 0 {
					t.Fatalf("empty shard state=%+v", st)
				}
			}
		}
		if !emptyFound {
			t.Fatal("empty shard status missing")
		}

		replay := getReplay(t, api.URL, resp.RequestID)
		if !replay.Query.Exact || !replay.Query.Complete {
			t.Fatalf("replay status wrong: %+v", replay.Query)
		}
		if len(replay.Events) == 0 || replay.Events[len(replay.Events)-1].Kind != "STOP_PROVED" {
			t.Fatalf("replay events lack stop proof: %+v", replay.Events)
		}
	})

	t.Run("bad_bound_returns_boundary_failure_and_unknown_incomplete", func(t *testing.T) {
		resp := postTopK(t, api.URL, topKPayload{Generation: string(gen), K: 1}, "bad_bound:0")
		if resp.Complete || resp.Exact || resp.Failure == nil {
			t.Fatalf("response=%+v", resp)
		}
		if resp.Failure.Class != string(domain.FailureBoundViolation) {
			t.Fatalf("class=%s want=%s", resp.Failure.Class, domain.FailureBoundViolation)
		}
	})

	t.Run("timeout_returns_timeout_known_bound_and_no_exact_claim", func(t *testing.T) {
		resp := postTopK(t, api.URL, topKPayload{Generation: string(gen), K: 5}, "slow:2")
		if resp.Complete || resp.Exact || resp.Failure == nil {
			t.Fatalf("response=%+v", resp)
		}
		if resp.Failure.Class != string(domain.FailureTimeout) {
			t.Fatalf("class=%s want=%s", resp.Failure.Class, domain.FailureTimeout)
		}
	})

	t.Run("generation_mismatch_is_rejected", func(t *testing.T) {
		resp := postTopK(t, api.URL, topKPayload{Generation: string(gen), K: 3}, "wrong_generation:1")
		if resp.Complete || resp.Exact || resp.Failure == nil {
			t.Fatalf("response=%+v", resp)
		}
		if resp.Failure.Class != string(domain.FailureGenerationMismatch) {
			t.Fatalf("class=%s want=%s", resp.Failure.Class, domain.FailureGenerationMismatch)
		}
	})
}

func postTopK(t *testing.T, base string, payload topKPayload, fault string) topKResponse {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, base+"/v1/top-k", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if fault != "" {
		req.Header.Set("X-Test-Fault", fault)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("HTTP %d", resp.StatusCode)
	}
	var out topKResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

func getReplay(t *testing.T, base, requestID string) replayResponse {
	t.Helper()
	resp, err := http.Get(base + "/v1/replay/" + requestID)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("replay HTTP %d", resp.StatusCode)
	}
	var out replayResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

func testLogger(t *testing.T) *log.Logger {
	return log.New(testWriter{t}, "IT ", log.LstdFlags|log.Lmicroseconds)
}

type testWriter struct{ t *testing.T }

func (w testWriter) Write(p []byte) (int, error) {
	w.t.Logf("%s", bytes.TrimSpace(p))
	return len(p), nil
}

func integrationDatabaseURL() string {
	if url := os.Getenv("TOPK_TEST_DATABASE_URL"); url != "" {
		return url
	}
	return "postgres://admin@/opp297_topk?host=/var/run/postgresql&sslmode=disable"
}

func cleanupGeneration(ctx context.Context, t *testing.T, dbURL string, gen domain.GenerationID) {
	t.Helper()
	conn, err := pgx.Connect(ctx, dbURL)
	if err != nil {
		t.Logf("cleanup connect: %v", err)
		return
	}
	defer conn.Close(ctx)
	statements := []string{
		`DELETE FROM topk_query_results WHERE request_id IN (SELECT request_id FROM topk_queries WHERE generation_id=$1)`,
		`DELETE FROM topk_query_shard_states WHERE request_id IN (SELECT request_id FROM topk_queries WHERE generation_id=$1)`,
		`DELETE FROM topk_query_events WHERE request_id IN (SELECT request_id FROM topk_queries WHERE generation_id=$1)`,
		`DELETE FROM topk_queries WHERE generation_id=$1`,
		`DELETE FROM items WHERE generation_id=$1`,
		`DELETE FROM generations WHERE generation_id=$1`,
	}
	for _, statement := range statements {
		if _, err := conn.Exec(ctx, statement, string(gen)); err != nil {
			t.Logf("cleanup %q: %v", statement, err)
		}
	}
}
