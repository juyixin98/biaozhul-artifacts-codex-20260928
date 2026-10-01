package kernel

import (
	"strings"
	"testing"

	"specbatch/internal/model"
)

func TestComputeDoublesAndStabilizesRecordOrder(t *testing.T) {
	records := []model.InputRecord{{ID: "z", Value: 3}, {ID: "a", Value: -2}}
	got, err := Compute("job", "shard", records)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].ID != "a" || got[0].Value != -4 || got[1].ID != "z" || got[1].Value != 6 {
		t.Fatalf("unexpected deterministic result: %+v", got)
	}
}

func TestValidateRejectsCrossShardArtifact(t *testing.T) {
	lease := model.Lease{
		JobID: "j", ShardID: "expected", AttemptNo: 1,
		Records: []model.InputRecord{{ID: "r1", Value: 4}},
	}
	doc := model.ArtifactDocument{
		SchemaVersion: "specbatch/artifact/v1", JobID: "j", ShardID: "other", AttemptNo: 1,
		Kernel: Name, KernelVersion: Version,
		Records: []model.OutputRecord{{ShardID: "other", ID: "r1", Value: 8}},
	}
	_, _, err := Validate(doc, lease)
	if err == nil || !strings.Contains(err.Error(), "identity mismatch") {
		t.Fatalf("expected identity mismatch, got %v", err)
	}
}

func TestValidateRejectsWrongComputedValue(t *testing.T) {
	lease := model.Lease{JobID: "j", ShardID: "s", AttemptNo: 1, Records: []model.InputRecord{{ID: "r1", Value: 4}}}
	doc := model.ArtifactDocument{
		SchemaVersion: "specbatch/artifact/v1", JobID: "j", ShardID: "s", AttemptNo: 1,
		Kernel: Name, KernelVersion: Version,
		Records: []model.OutputRecord{{ShardID: "s", ID: "r1", Value: 999}},
	}
	_, _, err := Validate(doc, lease)
	if err == nil || !strings.Contains(err.Error(), "record r1 mismatch") {
		t.Fatalf("expected computed-value mismatch, got %v", err)
	}
}
