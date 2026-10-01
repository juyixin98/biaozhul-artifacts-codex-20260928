package worker

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"opp300-speculative-batch/internal/participant"
	"opp300-speculative-batch/internal/protocol"
)

type fixtureShard struct {
	Ordinal                 int               `json:"ordinal"`
	Key                     string            `json:"key"`
	Input                   string            `json:"input"`
	Behavior                protocol.Behavior `json:"behavior"`
	ExpectedOutput          string            `json:"expected_output"`
	ExpectedFailureCategory string            `json:"expected_failure_category"`
}

type fixtureFile struct {
	IDempotencyKey string         `json:"idempotency_key"`
	InputRunID     string         `json:"input_run_id"`
	Version        string         `json:"version"`
	Shards         []fixtureShard `json:"shards"`
}

func TestKernelMatchesGoldenFixture(t *testing.T) {
	fixture := loadFixture(t)
	kernel := &Kernel{}
	for _, shard := range fixture.Shards {
		t.Run(shard.Key, func(t *testing.T) {
			result, err := kernel.Compute(shard.Input, shard.Behavior)
			if shard.ExpectedFailureCategory != "" {
				var failure *FailureError
				if !errors.As(err, &failure) {
					t.Fatalf("expected FailureError, got %v", err)
				}
				if failure.Category != shard.ExpectedFailureCategory {
					t.Fatalf("failure category = %q, want %q", failure.Category, shard.ExpectedFailureCategory)
				}
				return
			}
			if err != nil {
				t.Fatalf("Compute returned error: %v", err)
			}
			if result.Output != shard.ExpectedOutput {
				t.Fatalf("output = %q, want %q", result.Output, shard.ExpectedOutput)
			}
			if len(result.Output) != int(result.SizeBytes) {
				t.Fatalf("size = %d, want %d", result.SizeBytes, len(result.Output))
			}
			if len(result.ChecksumSHA256) != 64 {
				t.Fatalf("checksum length = %d, want 64", len(result.ChecksumSHA256))
			}
		})
	}
}

func TestKernelRejectsUnknownSuccessAndValidationFailure(t *testing.T) {
	kernel := &Kernel{}
	if _, err := kernel.Compute("   ", protocol.Behavior{}); err == nil {
		t.Fatal("empty input unexpectedly succeeded")
	} else {
		var failure *FailureError
		if !errors.As(err, &failure) || failure.Category != protocol.FailureValidation {
			t.Fatalf("error = %v, want VALIDATION failure", err)
		}
	}

	_, err := kernel.Compute("input", protocol.Behavior{Fail: true, FailureCategory: "WEIRD"})
	var failure *FailureError
	if !errors.As(err, &failure) {
		t.Fatalf("expected FailureError, got %v", err)
	}
	if failure.Category != protocol.FailureCompute {
		t.Fatalf("unknown category normalized to %q, want COMPUTE", failure.Category)
	}
}

func TestRunRequiresIdentityAndCallback(t *testing.T) {
	kernel := &Kernel{Client: participant.NewClient()}
	err := kernel.Run(context.Background(), participant.ExecuteRequest{})
	if err == nil {
		t.Fatal("Run unexpectedly accepted request without identity")
	}
	var failure *FailureError
	if !errors.As(err, &failure) || failure.Category != protocol.FailureValidation {
		t.Fatalf("error = %v, want VALIDATION", err)
	}
}

func loadFixture(t *testing.T) fixtureFile {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "fixtures", "minimal-job.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture fixtureFile
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if fixture.IDempotencyKey != "fixture-minimal-001" || fixture.Version != "local-1.0.0" {
		t.Fatalf("fixture identity is incomplete: %+v", fixture)
	}
	return fixture
}
