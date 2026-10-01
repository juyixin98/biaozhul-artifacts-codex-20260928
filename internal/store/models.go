package store

import (
	"encoding/json"
	"time"
)

type Job struct {
	ID          string
	Dim         int
	Status      string
	Aggregate   json.RawMessage
	Version     int64
	CreatedAt   time.Time
	FinalizedAt *time.Time
}

type Shard struct {
	JobID          string
	ShardID        string
	Seq            int64
	N              int64
	Weight         float64
	Summary        json.RawMessage
	SubmittedRunID string
	CreatedAt      time.Time
}

type Run struct {
	ID         string
	RunNumber  int64
	JobID      *string
	Route      string
	Method     string
	StatusCode int
	Status     string
	Request    json.RawMessage
	Error      json.RawMessage
	StartedAt  time.Time
	FinishedAt *time.Time
}

type RunEvent struct {
	RunID     string
	Index     int64
	Stage     string
	Result    string
	Reason    string
	State     json.RawMessage
	CreatedAt time.Time
}
