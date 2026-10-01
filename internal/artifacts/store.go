package artifacts

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

var (
	ErrChecksumMismatch = errors.New("artifact checksum mismatch")
	ErrInvalidChecksum  = errors.New("invalid artifact checksum")
)

type StagedArtifact struct {
	ID             string
	TemporaryPath  string
	ChecksumSHA256 string
	SizeBytes      int64
}

type Store struct {
	root string
}

func NewStore(root string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(root, "staging"), 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Join(root, "published"), 0o755); err != nil {
		return nil, err
	}
	return &Store{root: root}, nil
}

func (s *Store) Root() string {
	return s.root
}

func (s *Store) Stage(r io.Reader, expectedSHA256 string, maxBytes int64) (StagedArtifact, error) {
	expected, err := normalizeChecksum(expectedSHA256)
	if err != nil {
		return StagedArtifact{}, err
	}
	id, err := newID()
	if err != nil {
		return StagedArtifact{}, err
	}
	path := filepath.Join(s.root, "staging", id+".tmp")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return StagedArtifact{}, err
	}
	hash := sha256.New()
	limited := io.LimitReader(r, maxBytes+1)
	written, err := io.Copy(io.MultiWriter(file, hash), limited)
	closeErr := file.Close()
	if err != nil {
		_ = os.Remove(path)
		return StagedArtifact{}, err
	}
	if closeErr != nil {
		_ = os.Remove(path)
		return StagedArtifact{}, closeErr
	}
	if written > maxBytes {
		_ = os.Remove(path)
		return StagedArtifact{}, fmt.Errorf("artifact exceeds %d bytes", maxBytes)
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if actual != expected {
		_ = os.Remove(path)
		return StagedArtifact{}, fmt.Errorf("%w: expected %s, got %s", ErrChecksumMismatch, expected, actual)
	}
	return StagedArtifact{
		ID:             id,
		TemporaryPath:  path,
		ChecksumSHA256: actual,
		SizeBytes:      written,
	}, nil
}

type CommitInput struct {
	Staged StagedArtifact
	JobID  string
}

type CommittedArtifact struct {
	ID          string
	StoragePath string
}

func (s *Store) Commit(input CommitInput) (CommittedArtifact, error) {
	if _, err := os.Stat(input.Staged.TemporaryPath); err != nil {
		return CommittedArtifact{}, err
	}
	jobPart := safeName(input.JobID)
	shardPart := safeName(input.Staged.ChecksumSHA256[:12])
	dir := filepath.Join(s.root, "published", jobPart, shardPart)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return CommittedArtifact{}, err
	}
	finalName := safeName(input.Staged.ChecksumSHA256) + ".out"
	finalPath := filepath.Join(dir, finalName)
	if err := os.Link(input.Staged.TemporaryPath, finalPath); err != nil && !errors.Is(err, os.ErrExist) {
		return CommittedArtifact{}, err
	}
	if err := os.Chmod(finalPath, 0o440); err != nil && !errors.Is(err, os.ErrPermission) {
		return CommittedArtifact{}, err
	}
	rel, err := filepath.Rel(s.root, finalPath)
	if err != nil {
		return CommittedArtifact{}, err
	}
	return CommittedArtifact{ID: input.Staged.ID, StoragePath: rel}, nil
}

func (s *Store) RemoveStaged(staged StagedArtifact) {
	if staged.TemporaryPath == "" {
		return
	}
	_ = os.Remove(staged.TemporaryPath)
}

func (s *Store) ReadPublished(rel string) ([]byte, error) {
	clean := filepath.Clean(rel)
	if strings.HasPrefix(clean, "..") || filepath.IsAbs(clean) {
		return nil, fmt.Errorf("invalid published artifact path")
	}
	return os.ReadFile(filepath.Join(s.root, clean))
}

func (s *Store) CleanupStaging() error {
	staging := filepath.Join(s.root, "staging")
	entries, err := os.ReadDir(staging)
	if os.IsNotExist(err) {
		return os.MkdirAll(staging, 0o755)
	}
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if err := os.RemoveAll(filepath.Join(staging, entry.Name())); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) EnsureDirs() error {
	if _, err := NewStore(s.root); err != nil {
		return err
	}
	return nil
}

func normalizeChecksum(value string) (string, error) {
	if len(value) != 64 {
		return "", ErrInvalidChecksum
	}
	for _, r := range value {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f')) {
			return "", ErrInvalidChecksum
		}
	}
	return value, nil
}

func safeName(value string) string {
	value = filepath.Base(value)
	value = strings.NewReplacer("/", "_", "\\", "_", "..", "_").Replace(value)
	if value == "." || value == "" {
		return "unknown"
	}
	return value
}

func newID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return "art-" + hex.EncodeToString(b[:]), nil
}
