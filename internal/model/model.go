package model

import "time"

const (
	JobQueued    = "queued"
	JobRunning   = "running"
	JobSucceeded = "succeeded"
	JobFailed    = "failed"
)

const (
	ShardQueued    = "queued"
	ShardRunning   = "running"
	ShardSucceeded = "succeeded"
	ShardFailed    = "failed"
)

const (
	AttemptRunning   = "running"
	AttemptSucceeded = "succeeded"
	AttemptFailed    = "failed"
	AttemptRejected  = "rejected"
)

const (
	AttemptInitial     = "initial"
	AttemptRetry       = "retry"
	AttemptSpeculative = "speculative"
)

// Failure categories are stable strings used by tests and operational logs.
const (
	FailureSimulatedWorker       = "simulated_worker_failure"
	FailureValidation            = "artifact_validation_failed"
	FailureSupersededSubmission  = "superseded_late_submission"
	FailureCoordinatorRestart    = "coordinator_restart"
	FailureInvalidToken          = "invalid_attempt_token"
	FailureAttemptLimitExhausted = "attempt_limit_exhausted"
	FailureBadRequest            = "bad_request"
	FailureUnknown               = "unknown_worker_failure"
)

type Job struct {
	ID                   string
	Status               string
	TotalShards          int
	CompletedShards      int
	MaxAttemptsPerShard  int
	MaxSpeculativeCopies int
	SpecAfter            time.Duration
	CreatedAt            time.Time
	UpdatedAt            time.Time
}

type Shard struct {
	JobID     string
	ID        string
	Index     int
	Status    string
	CreatedAt time.Time
	UpdatedAt time.Time
}

type Attempt struct {
	JobID           string
	ShardID         string
	AttemptNo       int
	Token           string
	WorkerID        string
	Kind            string
	Status          string
	FailureCategory string
	Progress        float64
	StartedAt       time.Time
	LastHeartbeatAt time.Time
	EndedAt         *time.Time
}

type Artifact struct {
	JobID       string
	ShardID     string
	AttemptNo   int
	SHA256      string
	Size        int64
	PublishedAt time.Time
}

type Event struct {
	Seq       int64
	JobID     string
	ShardID   string
	AttemptNo int
	WorkerID  string
	Type      string
	Status    string
	Category  string
	Reason    string
	Payload   []byte
	CreatedAt time.Time
}

type Lease struct {
	JobID            string                 `json:"job_id"`
	ShardID          string                 `json:"shard_id"`
	AttemptNo        int                    `json:"attempt_no"`
	Token            string                 `json:"token"`
	Kind             string                 `json:"kind"`
	Records          []InputRecord          `json:"records"`
	ExpectedRecordID []string               `json:"expected_record_ids"`
	Behavior         map[string]int         `json:"behavior"`
	FailKinds        []string               `json:"fail_kinds"`
	Kernel           string                 `json:"kernel"`
	KernelVersion    string                 `json:"kernel_version"`
	ProtocolVersion  string                 `json:"protocol_version"`
	Extra            map[string]interface{} `json:"extra,omitempty"`
}

type InputRecord struct {
	ID    string `json:"id"`
	Value int    `json:"value"`
}

type OutputRecord struct {
	ShardID string `json:"shard_id"`
	ID      string `json:"id"`
	Value   int    `json:"value"`
}

type ArtifactDocument struct {
	SchemaVersion string         `json:"schema_version"`
	JobID         string         `json:"job_id"`
	ShardID       string         `json:"shard_id"`
	AttemptNo     int            `json:"attempt_no"`
	Kernel        string         `json:"kernel"`
	KernelVersion string         `json:"kernel_version"`
	Records       []OutputRecord `json:"records"`
}

type MergeDocument struct {
	SchemaVersion string         `json:"schema_version"`
	JobID         string         `json:"job_id"`
	RecordCount   int            `json:"record_count"`
	Records       []OutputRecord `json:"records"`
}

type Manifest struct {
	SchemaVersion        string          `json:"schema_version"`
	JobID                string          `json:"job_id"`
	Kernel               string          `json:"kernel"`
	KernelVersion        string          `json:"kernel_version"`
	ProtocolVersion      string          `json:"protocol_version"`
	MaxAttemptsPerShard  int             `json:"max_attempts_per_shard"`
	MaxSpeculativeCopies int             `json:"max_speculative_copies"`
	SpecAfterMillis      int             `json:"spec_after_ms"`
	Shards               []ManifestShard `json:"shards"`
}

type ManifestShard struct {
	ID        string                 `json:"id"`
	Records   []InputRecord          `json:"records"`
	Behavior  map[string]int         `json:"behavior"`
	FailKinds []string               `json:"fail_kinds"`
	Extra     map[string]interface{} `json:"extra,omitempty"`
}
