package store

import (
	"context"
	"crypto/rand"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"

	"github.com/lib/pq"

	"specbatch/internal/model"
)

//go:embed schema.sql
var schemaSQL string

var (
	ErrNotFound       = errors.New("not found")
	ErrConflict       = errors.New("conflict")
	ErrInvalidToken   = errors.New("invalid attempt token")
	ErrWrongEpoch     = errors.New("attempt belongs to a previous coordinator epoch")
	ErrValidation     = errors.New("artifact validation failed")
	ErrAttemptExpired = errors.New("attempt expired")
)

type Store struct{ db *sql.DB }

type Claim struct {
	Epoch int64
	Lease model.Lease
}

type SubmitInput struct {
	Token    string
	WorkerID string
	Document model.ArtifactDocument
	Content  []byte
}

type SubmitResult struct {
	Won       bool
	JobID     string
	ShardID   string
	AttemptNo int
	Epoch     int64
}

func Open(ctx context.Context, databaseURL string) (*Store, error) {
	db, err := sql.Open("postgres", databaseURL)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(8)
	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	s := &Store{db: db}
	return s, s.migrate(ctx)
}

func (s *Store) Close() error { return s.db.Close() }
func (s *Store) DB() *sql.DB  { return s.db }
func (s *Store) migrate(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, schemaSQL)
	return err
}

func randomToken() (string, error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func isUnique(err error) bool {
	var pqErr *pq.Error
	return errors.As(err, &pqErr) && pqErr.Code == "23505"
}
