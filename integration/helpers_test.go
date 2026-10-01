//go:build integration

package integration

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/protocol"
)

type speculativeFixture struct {
	IDempotencyKey       string              `json:"idempotency_key"`
	InputRunID           string              `json:"input_run_id"`
	Version              string              `json:"version"`
	ExpectedOutput       string              `json:"expected_output"`
	ExpectedFailureClass string              `json:"expected_failure_category"`
	Shard                protocol.ShardInput `json:"shard"`
}

type raceFixture struct {
	IDempotencyKey string              `json:"idempotency_key"`
	InputRunID     string              `json:"input_run_id"`
	Version        string              `json:"version"`
	ExpectedOutput string              `json:"expected_output"`
	Shard          protocol.ShardInput `json:"shard"`
}

func loadSpeculativeFixture(t *testing.T) speculativeFixture {
	var fixture speculativeFixture
	loadJSONFixture(t, "speculative-job.json", &fixture)
	return fixture
}

func loadRestartFixture(t *testing.T) speculativeFixture {
	var fixture speculativeFixture
	loadJSONFixture(t, "restart-job.json", &fixture)
	return fixture
}

func loadRaceFixture(t *testing.T) raceFixture {
	var fixture raceFixture
	loadJSONFixture(t, "race-job.json", &fixture)
	return fixture
}

func loadJSONFixture(t *testing.T, name string, dst any) {
	t.Helper()
	path := filepath.Join("..", "testdata", "fixtures", name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatalf("decode fixture %s: %v", name, err)
	}
}

func unique(prefix string) string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return prefix + "-" + hex.EncodeToString(b[:])
}

func assertEventType(t *testing.T, events []protocol.Event, eventType string) protocol.Event {
	t.Helper()
	for _, event := range events {
		if event.Type == eventType {
			return event
		}
	}
	t.Fatalf("event type %q not found in %d events", eventType, len(events))
	return protocol.Event{}
}

func eventHas(events []protocol.Event, eventType, key, value string) bool {
	for _, event := range events {
		if event.Type != eventType {
			continue
		}
		got, _ := event.Payload[key].(string)
		if got == value {
			return true
		}
	}
	return false
}

func countFiles(t *testing.T, root string) int {
	t.Helper()
	count := 0
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			return err
		}
		if !entry.IsDir() {
			count++
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return count
}

func waitForAttemptState(t *testing.T, ctx context.Context, h *harness, jobID string, attemptNumber int, state string) {
	t.Helper()
	deadline, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	for {
		attempts := h.attempts(deadline, jobID)
		for _, attempt := range attempts {
			if attempt.AttemptNumber == attemptNumber && attempt.State == state {
				return
			}
		}
		select {
		case <-deadline.Done():
			t.Fatalf("attempt %d never reached %s, got %+v", attemptNumber, state, attempts)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func waitForAttemptCount(t *testing.T, ctx context.Context, h *harness, jobID string, count int) {
	t.Helper()
	deadline, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	for {
		attempts := h.attempts(deadline, jobID)
		if len(attempts) >= count {
			return
		}
		select {
		case <-deadline.Done():
			t.Fatalf("attempt count never reached %d, got %+v", count, attempts)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func stageArtifact(ctx context.Context, baseURL, attemptID, epoch, checksum string, payload []byte) (participant.StageResponse, error) {
	endpoint, err := url.Parse(baseURL + "/v1/attempts/" + attemptID + "/artifacts")
	if err != nil {
		return participant.StageResponse{}, err
	}
	query := endpoint.Query()
	query.Set("checksum_sha256", checksum)
	query.Set("epoch", epoch)
	endpoint.RawQuery = query.Encode()

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(payload))
	if err != nil {
		return participant.StageResponse{}, err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return participant.StageResponse{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(response.Body)
		return participant.StageResponse{}, fmt.Errorf("stage status %d: %s", response.StatusCode, string(body))
	}
	var staged participant.StageResponse
	return staged, json.NewDecoder(response.Body).Decode(&staged)
}

func completeAttempt(ctx context.Context, t *testing.T, baseURL, attemptID string, body participant.CompleteRequest) (int, string) {
	t.Helper()
	payload, _ := json.Marshal(body)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost,
		baseURL+"/v1/attempts/"+attemptID+"/complete", strings.NewReader(string(payload)))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	data, _ := io.ReadAll(response.Body)
	return response.StatusCode, string(data)
}
