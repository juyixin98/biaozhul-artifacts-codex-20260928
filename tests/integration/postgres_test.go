package integration

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"topk-demo/internal/coordinator"
	"topk-demo/internal/fixture"
	"topk-demo/internal/protocol"
	"topk-demo/internal/seed"
	"topk-demo/internal/shardserver"
	"topk-demo/internal/shardstore"
	"topk-demo/internal/state"
)

func TestPostgresTopKAndReplay(t *testing.T) {
	dsn := prepareDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dataStore, err := shardstore.NewPostgresStore(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer dataStore.Close()
	runStore, err := state.NewPostgresStore(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer runStore.Close()

	shardHTTP := httptest.NewServer(shardserver.New(dataStore, nil).Routes())
	defer shardHTTP.Close()
	coordCfg := coordinator.Config{
		Generation:   fixture.Generation,
		DefaultK:     5,
		DefaultPage:  4,
		ShardTimeout: time.Second,
	}
	for _, id := range fixture.ShardIDs() {
		coordCfg.Shards = append(coordCfg.Shards, coordinator.ShardConfig{ID: id, URL: shardHTTP.URL})
	}
	coordHTTP := httptest.NewServer(coordinator.New(coordCfg, runStore, nil).Routes())
	defer coordHTTP.Close()

	body := `{"k":5,"page_size":4}`
	resp := postJSON(t, coordHTTP.URL+"/v1/topk", body)
	if resp.StatusCode != 200 {
		t.Fatalf("topk status = %d", resp.StatusCode)
	}
	var query protocol.QueryResponse
	decodeBody(t, resp, &query)
	if !query.Exact || !query.Complete || query.StopReason != protocol.ReasonExactThreshold {
		t.Fatalf("query should be exact: exact=%v complete=%v reason=%s errors=%+v", query.Exact, query.Complete, query.StopReason, query.Errors)
	}
	assertCandidates(t, query.Results, fixture.ExpectedTop5())
	if query.Metrics.CandidatesReturned >= int64(len(fixture.Items())) {
		t.Fatalf("expected early termination, returned=%d total=%d", query.Metrics.CandidatesReturned, len(fixture.Items()))
	}

	replayResp := get(t, coordHTTP.URL+"/v1/runs/"+query.RunID)
	if replayResp.StatusCode != 200 {
		t.Fatalf("replay status = %d", replayResp.StatusCode)
	}
	var replay struct {
		Run    protocol.RunRecord     `json:"run"`
		Events []protocol.EventRecord `json:"events"`
	}
	decodeBody(t, replayResp, &replay)
	if !replay.Run.Exact || replay.Run.Generation != fixture.Generation {
		t.Fatalf("replayed run has wrong state: %+v", replay.Run)
	}
	eventTypes := map[string]protocol.EventRecord{}
	for _, event := range replay.Events {
		eventTypes[event.EventType] = event
		if event.Generation != fixture.Generation {
			t.Fatalf("event %s has wrong generation %q", event.EventType, event.Generation)
		}
	}
	for _, required := range []string{"query_start", "initial_scan_received", "query_stop"} {
		if _, ok := eventTypes[required]; !ok {
			t.Fatalf("replay missing diagnostic event %q; got %d events", required, len(replay.Events))
		}
	}
	if !eventTypes["query_stop"].Exact {
		t.Fatalf("stop event must explain exact acceptance: %+v", eventTypes["query_stop"])
	}

	staleResp := postJSON(t, coordHTTP.URL+"/v1/topk", `{"generation":"gen-stale","k":5}`)
	if staleResp.StatusCode != 409 {
		t.Fatalf("stale generation status = %d, want 409", staleResp.StatusCode)
	}

	staleShard := postJSON(t, shardHTTP.URL+"/v1/shards/scan", `{"generation":"gen-stale","shard_id":"shard-a","page_size":2}`)
	if staleShard.StatusCode != 409 {
		t.Fatalf("stale shard generation status = %d, want 409", staleShard.StatusCode)
	}
}

func prepareDatabase(t *testing.T) string {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping PostgreSQL integration test in -short mode")
	}
	adminDSN := firstWorkingDSN()
	ctx := context.Background()
	admin, err := pgx.Connect(ctx, adminDSN)
	if err != nil {
		t.Skipf("PostgreSQL is not available: %v", err)
	}
	defer admin.Close(ctx)

	var suffix [6]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		t.Fatal(err)
	}
	name := "topk_demo_it_" + hex.EncodeToString(suffix[:])
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+name); err != nil {
		t.Skipf("cannot create integration database: %v", err)
	}
	t.Cleanup(func() {
		_, _ = admin.Exec(context.Background(), "DROP DATABASE "+name)
	})

	dsn := "host=/var/run/postgresql user=admin dbname=" + name + " sslmode=disable"
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	schema, err := os.ReadFile("../../db/schema.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, string(schema)); err != nil {
		t.Fatal(err)
	}
	if err := seed.Run(ctx, dsn); err != nil {
		t.Fatal(err)
	}
	return dsn
}

func firstWorkingDSN() string {
	if value := os.Getenv("TOPK_TEST_DSN"); value != "" {
		return value
	}
	return "host=/var/run/postgresql user=admin dbname=postgres sslmode=disable"
}

func postJSON(t *testing.T, url, body string) *httpResponse {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return readResponse(t, resp)
}

func get(t *testing.T, url string) *httpResponse {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	return readResponse(t, resp)
}

type httpResponse struct {
	StatusCode int
	Body       []byte
}

func readResponse(t *testing.T, resp *http.Response) *httpResponse {
	t.Helper()
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return &httpResponse{StatusCode: resp.StatusCode, Body: data}
}

func decodeBody(t *testing.T, resp *httpResponse, dst any) {
	t.Helper()
	if err := json.Unmarshal(resp.Body, dst); err != nil {
		t.Fatalf("decode response: %v body=%s", err, string(resp.Body))
	}
}

func assertCandidates(t *testing.T, got, want []protocol.Candidate) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d want %d got=%+v", len(got), len(want), got)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("candidate[%d] = %+v want %+v full=%+v", i, got[i], want[i], got)
		}
	}
}
