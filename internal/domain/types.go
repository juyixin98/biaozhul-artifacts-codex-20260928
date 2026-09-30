package domain

import (
	"errors"
	"fmt"
)

// GenerationID binds every candidate and every shard page to one data world.
type GenerationID string

type ItemID string

// Key is the globally agreed ranking. Better returns true when k ranks above o.
type Key struct {
	Score int64  `json:"score"`
	ID    ItemID `json:"id"`
}

func (k Key) Better(o Key) bool {
	if k.Score != o.Score {
		return k.Score > o.Score
	}
	return k.ID < o.ID
}

func (k Key) Equal(o Key) bool { return k == o }

func (k Key) String() string { return fmt.Sprintf("score=%d,id=%q", k.Score, k.ID) }

type Candidate struct {
	ID    ItemID `json:"id"`
	Score int64  `json:"score"`
}

func (c Candidate) Key() Key { return Key{Score: c.Score, ID: c.ID} }

// PageRequest is the protocol used by the coordinator to read one ordered page.
type PageRequest struct {
	Generation GenerationID `json:"generation"`
	ShardID    int          `json:"shard_id"`
	Limit      int          `json:"limit"`
	Cursor     *Key         `json:"cursor,omitempty"`
}

// PageResponse is authoritative for the keys it contains. UpperBound is the
// best key that may still occur after Cursor. For a non-final page it is
// authenticated from the p+1 probe rather than taken on trust.
type PageResponse struct {
	Generation GenerationID `json:"generation"`
	ShardID    int          `json:"shard_id"`
	Items      []Candidate  `json:"items"`
	HasMore    bool         `json:"has_more"`
	UpperBound *Key         `json:"upper_bound,omitempty"`
	Cursor     *Key         `json:"cursor,omitempty"`
	BoundProof string       `json:"bound_proof,omitempty"`
	RowsRead   int          `json:"rows_read"`
}

type FailureClass string

const (
	FailureNone               FailureClass = ""
	FailureTimeout            FailureClass = "SHARD_TIMEOUT"
	FailureProtocol           FailureClass = "SHARD_PROTOCOL"
	FailureBoundViolation     FailureClass = "BOUND_VIOLATION"
	FailureGenerationMismatch FailureClass = "GENERATION_MISMATCH"
)

var ErrUnavailable = errors.New("shard participant unavailable")

type ShardError struct {
	Shard int
	Class FailureClass
	Err   error
}

func (e *ShardError) Error() string {
	return fmt.Sprintf("shard %d: %s: %v", e.Shard, e.Class, e.Err)
}

func (e *ShardError) Unwrap() error { return e.Err }
