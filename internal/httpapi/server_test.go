package httpapi_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"opp299/internal/config"
	"opp299/internal/httpapi"
	"opp299/internal/protocol"
	"opp299/internal/source"
	"opp299/internal/storage"
)

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	cfg := config.Default()
	cfg.DefaultBackpressureLimit = 8
	server := httpapi.New(cfg, storage.NewMemoryStore())
	return httptest.NewServer(server.Routes())
}

func httpJSON(t *testing.T, method, url, requestID string, body any) (int, http.Header, map[string]any) {
	t.Helper()
	data, _ := json.Marshal(body)
	req, err := http.NewRequest(method, url, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if requestID != "" {
		req.Header.Set("X-Request-ID", requestID)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var decoded map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, resp.Header, decoded
}

func TestHTTPRunExplainableResultAndFailure(t *testing.T) {
	server := newTestServer(t)
	defer server.Close()
	workload := source.Workload{
		TaskID: "http-task",
		Streams: []source.StreamSpec{
			{ID: protocol.StreamA, Events: []source.Event{
				{Kind: "record", Record: &protocol.Record{Stream: protocol.StreamA, Key: "k1", Seq: 1, Value: 2}},
				{Kind: "barrier", Barrier: &protocol.Barrier{ID: 1, Stream: protocol.StreamA, After: 1}},
			}},
			{ID: protocol.StreamB, Events: []source.Event{
				{Kind: "record", Record: &protocol.Record{Stream: protocol.StreamB, Key: "k1", Seq: 1, Value: 7}},
				{Kind: "barrier", Barrier: &protocol.Barrier{ID: 1, Stream: protocol.StreamB, After: 1}},
			}},
		},
	}
	status, header, put := httpJSON(t, http.MethodPut, server.URL+"/v1/tasks/http-task/workload", "req-workload", workload)
	if status != http.StatusOK || header.Get("X-Request-ID") != "req-workload" {
		t.Fatalf("PUT status=%d header=%q body=%v", status, header.Get("X-Request-ID"), put)
	}

	status, header, run := httpJSON(t, http.MethodPost, server.URL+"/v1/tasks/http-task/runs", "req-run", map[string]any{
		"mode":                       "unaligned",
		"backpressure_limit":         8,
		"max_concurrent_checkpoints": 1,
	})
	if status != http.StatusOK || header.Get("X-Request-ID") != "req-run" {
		t.Fatalf("run status=%d header=%s body=%v", status, header.Get("X-Request-ID"), run)
	}
	if run["request_id"] != "req-run" || len(run["steps"].([]any)) != 3 {
		t.Fatalf("run response is missing identity or steps: %v", run)
	}
	state := run["result"].(map[string]any)["state"].(map[string]any)
	counts := state["counts"].(map[string]any)
	k1 := counts["k1"].(map[string]any)
	if k1["a"].(float64) != 2 || k1["b"].(float64) != 7 {
		t.Fatalf("reference counts = %v, want A=2 B=7", k1)
	}

	status, _, bad := httpJSON(t, http.MethodPost, server.URL+"/v1/tasks/http-task/runs", "req-bad", map[string]any{
		"mode":                       "sideways",
		"backpressure_limit":         8,
		"max_concurrent_checkpoints": 1,
	})
	if status != http.StatusBadRequest || bad["category"] != "protocol_violation" {
		t.Fatalf("bad mode status=%d body=%v", status, bad)
	}
	if bad["request_id"] != "req-bad" {
		t.Fatalf("error body identity = %v", bad["request_id"])
	}

	status, _, cp := httpJSON(t, http.MethodGet, server.URL+"/v1/tasks/http-task/checkpoints/latest", "req-cp", nil)
	if status != http.StatusOK || cp["mode"] != "unaligned" {
		t.Fatalf("checkpoint status=%d body=%v", status, cp)
	}
}
