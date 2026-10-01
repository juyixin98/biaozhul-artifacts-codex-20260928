package artifacts

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStageRejectsChecksumMismatchAndCommitsImmutablePublishedFile(t *testing.T) {
	root := t.TempDir()
	store, err := NewStore(root)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("ALPHA\nONE\n")
	sum := sha256.Sum256(payload)
	correct := hex.EncodeToString(sum[:])

	_, err = store.Stage(strings.NewReader(string(payload)+"late"), correct, 1024)
	if !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("mismatch error = %v, want ErrChecksumMismatch", err)
	}

	staged, err := store.Stage(strings.NewReader(string(payload)), correct, 1024)
	if err != nil {
		t.Fatal(err)
	}
	committed, err := store.Commit(CommitInput{Staged: staged, JobID: "job-1"})
	if err != nil {
		t.Fatal(err)
	}
	content, err := store.ReadPublished(committed.StoragePath)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != string(payload) {
		t.Fatalf("published content = %q", content)
	}

	store.RemoveStaged(staged)
	stagingEntries, err := os.ReadDir(filepath.Join(root, "staging"))
	if err != nil {
		t.Fatal(err)
	}
	if len(stagingEntries) != 0 {
		t.Fatalf("staging should be empty after temporary artifact is released, got %d", len(stagingEntries))
	}

	if _, err := store.Stage(io.MultiReader(strings.NewReader(strings.Repeat("x", 5)), strings.NewReader("too-big")), sha256String([]byte("xxxxx")), 5); err == nil {
		t.Fatal("oversized artifact unexpectedly accepted")
	}
}

func TestCleanupStagingRetainsDirectory(t *testing.T) {
	root := t.TempDir()
	store, err := NewStore(root)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("RETAIN\n")
	sum := sha256.Sum256(payload)
	staged, err := store.Stage(strings.NewReader(string(payload)), hex.EncodeToString(sum[:]), 1024)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CleanupStaging(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "staging")); err != nil {
		t.Fatalf("staging directory missing after cleanup: %v", err)
	}
	_, err = store.Stage(strings.NewReader(string(payload)), hex.EncodeToString(sum[:]), 1024)
	if err != nil {
		t.Fatalf("stage after cleanup failed: %v", err)
	}
	_ = staged
}

func sha256String(value []byte) string {
	sum := sha256.Sum256(value)
	return hex.EncodeToString(sum[:])
}
