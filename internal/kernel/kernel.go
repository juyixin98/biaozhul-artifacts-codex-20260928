package kernel

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"

	"specbatch/internal/model"
)

const (
	Name    = "synthetic-double"
	Version = "v1.0.0"
)

var ErrInvalidDocument = errors.New("invalid artifact document")

// Compute doubles every synthetic input record deterministically. It is intentionally
// small but it validates identity and shard ownership so an artifact cannot be
// replayed against a different logical partition.
func Compute(jobID, shardID string, records []model.InputRecord) ([]model.OutputRecord, error) {
	if jobID == "" || shardID == "" {
		return nil, fmt.Errorf("%w: missing job or shard identity", ErrInvalidDocument)
	}
	seen := make(map[string]struct{}, len(records))
	out := make([]model.OutputRecord, 0, len(records))
	for _, rec := range records {
		if rec.ID == "" {
			return nil, fmt.Errorf("%w: empty record id", ErrInvalidDocument)
		}
		if _, ok := seen[rec.ID]; ok {
			return nil, fmt.Errorf("%w: duplicate record id %s", ErrInvalidDocument, rec.ID)
		}
		seen[rec.ID] = struct{}{}
		out = append(out, model.OutputRecord{ShardID: shardID, ID: rec.ID, Value: rec.Value * 2})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func Validate(doc model.ArtifactDocument, lease model.Lease) ([]model.OutputRecord, string, error) {
	if doc.SchemaVersion != "specbatch/artifact/v1" || doc.Kernel != Name || doc.KernelVersion != Version {
		return nil, "", fmt.Errorf("%w: unsupported artifact or kernel version", ErrInvalidDocument)
	}
	if doc.JobID != lease.JobID || doc.ShardID != lease.ShardID || doc.AttemptNo != lease.AttemptNo {
		return nil, "", fmt.Errorf("%w: attempt identity mismatch", ErrInvalidDocument)
	}
	if len(doc.Records) != len(lease.Records) {
		return nil, "", fmt.Errorf("%w: expected %d records, got %d", ErrInvalidDocument, len(lease.Records), len(doc.Records))
	}
	expected, err := Compute(lease.JobID, lease.ShardID, lease.Records)
	if err != nil {
		return nil, "", err
	}
	got := append([]model.OutputRecord(nil), doc.Records...)
	sort.Slice(got, func(i, j int) bool {
		if got[i].ID != got[j].ID {
			return got[i].ID < got[j].ID
		}
		return got[i].ShardID < got[j].ShardID
	})
	sort.Slice(expected, func(i, j int) bool {
		if expected[i].ID != expected[j].ID {
			return expected[i].ID < expected[j].ID
		}
		return expected[i].ShardID < expected[j].ShardID
	})
	if len(got) != len(expected) {
		return nil, "", fmt.Errorf("%w: record count mismatch", ErrInvalidDocument)
	}
	for i := range got {
		if got[i] != expected[i] {
			return nil, "", fmt.Errorf("%w: record %s mismatch", ErrInvalidDocument, expected[i].ID)
		}
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s|%s|%d|%d", doc.JobID, doc.ShardID, doc.AttemptNo, len(doc.Records))))
	return got, hex.EncodeToString(sum[:]), nil
}

func CanonicalArtifact(lease model.Lease) (model.ArtifactDocument, error) {
	records, err := Compute(lease.JobID, lease.ShardID, lease.Records)
	if err != nil {
		return model.ArtifactDocument{}, err
	}
	return model.ArtifactDocument{
		SchemaVersion: "specbatch/artifact/v1",
		JobID:         lease.JobID,
		ShardID:       lease.ShardID,
		AttemptNo:     lease.AttemptNo,
		Kernel:        Name,
		KernelVersion: Version,
		Records:       records,
	}, nil
}
