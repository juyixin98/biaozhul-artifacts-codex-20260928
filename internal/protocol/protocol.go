package protocol

import (
	"errors"
	"strconv"
	"time"
)

const (
	ReasonExactThreshold = "exact_threshold"
	ReasonExactExhausted = "exact_exhausted"
	ReasonIncomplete     = "incomplete_shard_failure"
	ReasonInvalid        = "invalid_shard_response"
)

const (
	FailureNone           = ""
	FailureTimeout        = "timeout"
	FailureNetwork        = "network_error"
	FailureHTTPStatus     = "http_status_error"
	FailureBadGeneration  = "bad_generation"
	FailureInvalidPayload = "invalid_payload"
	FailureInvalidShard   = "invalid_shard"
)

type Candidate struct {
	ItemID string `json:"item_id"`
	Score  int64  `json:"score"`
}

type BoundRequest struct {
	Generation string `json:"generation"`
	ShardID    string `json:"shard_id"`
	Cursor     string `json:"cursor,omitempty"`
}

type BoundResponse struct {
	Generation  string     `json:"generation"`
	ShardID     string     `json:"shard_id"`
	Cursor      string     `json:"cursor"`
	Exhausted   bool       `json:"exhausted"`
	Upper       *Candidate `json:"upper,omitempty"`
	ReadRows    int64      `json:"read_rows"`
	BoundProbes int64      `json:"bound_probes"`
}

type ScanRequest struct {
	Generation string `json:"generation"`
	ShardID    string `json:"shard_id"`
	Cursor     string `json:"cursor,omitempty"`
	PageSize   int    `json:"page_size"`
}

type ScanResponse struct {
	Generation         string      `json:"generation"`
	ShardID            string      `json:"shard_id"`
	Cursor             string      `json:"cursor"`
	NextCursor         string      `json:"next_cursor"`
	NextCursorAt       *Candidate  `json:"next_cursor_at,omitempty"`
	Exhausted          bool        `json:"exhausted"`
	Items              []Candidate `json:"items"`
	RequestedUpper     *Candidate  `json:"requested_upper,omitempty"`
	NextUpper          *Candidate  `json:"next_upper,omitempty"`
	ReadRows           int64       `json:"read_rows"`
	ReturnedRows       int64       `json:"returned_rows"`
	BoundProbes        int64       `json:"bound_probes"`
	ItemsIncludedUpper bool        `json:"items_included_upper"`
}

type QueryRequest struct {
	Generation     string `json:"generation,omitempty"`
	K              int    `json:"k"`
	PageSize       int    `json:"page_size,omitempty"`
	ShardTimeoutMS int    `json:"shard_timeout_ms,omitempty"`
}

type ShardProof struct {
	ShardID            string     `json:"shard_id"`
	State              string     `json:"state"`
	FailureCategory    string     `json:"failure_category,omitempty"`
	FailureMessage     string     `json:"failure_message,omitempty"`
	CursorExhausted    bool       `json:"cursor_exhausted"`
	StopUpper          *Candidate `json:"stop_upper,omitempty"`
	PagesScanned       int64      `json:"pages_scanned"`
	BoundProbes        int64      `json:"bound_probes"`
	StorageRowsRead    int64      `json:"storage_rows_read"`
	CandidatesReturned int64      `json:"candidates_returned"`
	AcceptedAs         string     `json:"accepted_as,omitempty"`
	RejectedReason     string     `json:"rejected_reason,omitempty"`
}

type QueryResponse struct {
	RunID      string       `json:"run_id"`
	Generation string       `json:"generation"`
	K          int          `json:"k"`
	Exact      bool         `json:"exact"`
	Complete   bool         `json:"complete"`
	StopReason string       `json:"stop_reason"`
	Results    []Candidate  `json:"results"`
	Threshold  *Candidate   `json:"threshold,omitempty"`
	Metrics    QueryMetrics `json:"metrics"`
	Proofs     []ShardProof `json:"proofs"`
	Errors     []ShardError `json:"errors,omitempty"`
}

type QueryMetrics struct {
	ShardsConfigured     int   `json:"shards_configured"`
	ShardsSucceeded      int   `json:"shards_succeeded"`
	ShardsFailed         int   `json:"shards_failed"`
	PagesScanned         int64 `json:"pages_scanned"`
	BoundProbes          int64 `json:"bound_probes"`
	StorageRowsRead      int64 `json:"storage_rows_read"`
	CandidatesReturned   int64 `json:"candidates_returned"`
	CandidatesConsidered int64 `json:"candidates_considered"`
}

type ShardError struct {
	ShardID  string `json:"shard_id"`
	Category string `json:"category"`
	Message  string `json:"message"`
}

type RunRecord struct {
	RunID                string
	Generation           string
	K                    int
	Complete             bool
	Exact                bool
	ResultCount          int
	StopReason           string
	BoundProbes          int64
	StorageRowsRead      int64
	CandidatesReturned   int64
	CandidatesConsidered int64
	ErrorSummary         string
	CreatedAt            time.Time
}

type EventRecord struct {
	RunID           string
	Sequence        int64
	EventType       string
	ShardID         string
	Generation      string
	Complete        bool
	Exact           bool
	HeapSize        int
	ThresholdScore  *int64
	ThresholdItemID string
	UpperScore      *int64
	UpperItemID     string
	CursorExhausted bool
	PagesScanned    int64
	BoundProbes     int64
	RowsRead        int64
	FailureCategory string
	Message         string
	CreatedAt       time.Time
}

// Better defines the only global ranking used by every shard, coordinator,
// fixture and test: higher score first, then lexicographically smaller ID.
func Better(a, b Candidate) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	return a.ItemID < b.ItemID
}

func LessKey(score int64, itemID string, afterScore int64, afterItemID string) bool {
	if score != afterScore {
		return score < afterScore
	}
	return itemID > afterItemID
}

func ValidateCandidate(c Candidate) error {
	if c.ItemID == "" {
		return errors.New("candidate item_id is required")
	}
	return nil
}

func DurationFromMillis(ms int, fallback time.Duration) time.Duration {
	if ms <= 0 {
		return fallback
	}
	return time.Duration(ms) * time.Millisecond
}

func CandidateKey(c *Candidate) string {
	if c == nil {
		return ""
	}
	return strconv.FormatInt(c.Score, 10) + "/" + c.ItemID
}
