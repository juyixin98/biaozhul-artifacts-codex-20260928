package protocol

import "time"

const (
	JobPending   = "pending"
	JobRunning   = "running"
	JobSucceeded = "succeeded"
	JobFailed    = "failed"

	ShardPending   = "pending"
	ShardRunning   = "running"
	ShardSucceeded = "succeeded"
	ShardFailed    = "failed"

	AttemptPending     = "pending"
	AttemptRunning     = "running"
	AttemptSucceeded   = "succeeded"
	AttemptFailed      = "failed"
	AttemptExpired     = "expired"
	AttemptInterrupted = "interrupted"
	AttemptSuperseded  = "superseded"

	FailureValidation   = "VALIDATION"
	FailureCompute      = "COMPUTE"
	FailureWorker       = "WORKER_UNAVAILABLE"
	FailureLeaseExpired = "LEASE_EXPIRED"
	FailureInterrupted  = "INTERRUPTED"
	FailureSuperseded   = "SUPERSEDED"
)

type ShardInput struct {
	Ordinal  int      `json:"ordinal"`
	Key      string   `json:"key"`
	Input    string   `json:"input"`
	Behavior Behavior `json:"behavior"`
}

type Behavior struct {
	DelayMS            int    `json:"delay_ms,omitempty"`
	OriginalDelayMS    int    `json:"original_delay_ms,omitempty"`
	SpeculativeDelayMS int    `json:"speculative_delay_ms,omitempty"`
	Fail               bool   `json:"fail,omitempty"`
	FailSpeculative    bool   `json:"fail_speculative,omitempty"`
	FailureCategory    string `json:"failure_category,omitempty"`
	FailureMessage     string `json:"failure_message,omitempty"`
}

type Policy struct {
	MaxAttemptsPerShard int `json:"max_attempts_per_shard"`
	MaxActiveAttempts   int `json:"max_active_attempts"`
	LeaseMS             int `json:"lease_ms"`
	SpeculateAfterMS    int `json:"speculate_after_ms"`
	HeartbeatIntervalMS int `json:"heartbeat_interval_ms"`
}

type JobSpec struct {
	IDempotencyKey string       `json:"idempotency_key"`
	Shards         []ShardInput `json:"shards"`
	Policy         Policy       `json:"policy"`
}

type Job struct {
	ID              string    `json:"id"`
	IDempotencyKey  string    `json:"idempotency_key"`
	State           string    `json:"state"`
	Version         int       `json:"version"`
	TotalShards     int       `json:"total_shards"`
	FailureCategory string    `json:"failure_category,omitempty"`
	FailureMessage  string    `json:"failure_message,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

type Shard struct {
	ID              string    `json:"id"`
	JobID           string    `json:"job_id"`
	Ordinal         int       `json:"ordinal"`
	Key             string    `json:"key"`
	Input           string    `json:"input"`
	Behavior        Behavior  `json:"behavior"`
	State           string    `json:"state"`
	AttemptNumber   int       `json:"attempt_number"`
	AttemptCount    int       `json:"attempt_count"`
	FailureCategory string    `json:"failure_category,omitempty"`
	FailureMessage  string    `json:"failure_message,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

type Attempt struct {
	ID              string     `json:"id"`
	JobID           string     `json:"job_id"`
	ShardID         string     `json:"shard_id"`
	AttemptNumber   int        `json:"attempt_number"`
	Kind            string     `json:"kind"`
	State           string     `json:"state"`
	Epoch           string     `json:"epoch"`
	WorkerURL       string     `json:"worker_url,omitempty"`
	Reason          string     `json:"reason,omitempty"`
	FailureCategory string     `json:"failure_category,omitempty"`
	FailureMessage  string     `json:"failure_message,omitempty"`
	ArtifactID      *string    `json:"artifact_id,omitempty"`
	ChecksumSHA256  string     `json:"checksum_sha256,omitempty"`
	SizeBytes       int64      `json:"size_bytes,omitempty"`
	LeaseExpiresAt  *time.Time `json:"lease_expires_at,omitempty"`
	StartedAt       *time.Time `json:"started_at,omitempty"`
	HeartbeatAt     *time.Time `json:"heartbeat_at,omitempty"`
	FinishedAt      *time.Time `json:"finished_at,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at"`
}

type Artifact struct {
	ID             string    `json:"id"`
	JobID          string    `json:"job_id"`
	ShardID        string    `json:"shard_id"`
	AttemptID      string    `json:"attempt_id"`
	StoragePath    string    `json:"storage_path"`
	ChecksumSHA256 string    `json:"checksum_sha256"`
	SizeBytes      int64     `json:"size_bytes"`
	CreatedAt      time.Time `json:"created_at"`
}

type Event struct {
	ID        int64          `json:"id"`
	JobID     string         `json:"job_id"`
	ShardID   string         `json:"shard_id,omitempty"`
	AttemptID string         `json:"attempt_id,omitempty"`
	Type      string         `json:"type"`
	Version   int            `json:"version"`
	Epoch     string         `json:"epoch,omitempty"`
	Payload   map[string]any `json:"payload,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
}

type Progress struct {
	State           string `json:"state"`
	Version         int    `json:"version"`
	TotalShards     int    `json:"total_shards"`
	Succeeded       int    `json:"succeeded"`
	Failed          int    `json:"failed"`
	Pending         int    `json:"pending"`
	Running         int    `json:"running"`
	ActiveAttempts  int    `json:"active_attempts"`
	AttemptsCreated int    `json:"attempts_created"`
}

type BudgetDecision struct {
	Allowed bool   `json:"allowed"`
	Reason  string `json:"reason"`
}

type PublishedShardResult struct {
	Ordinal        int    `json:"ordinal"`
	Key            string `json:"key"`
	AttemptID      string `json:"attempt_id"`
	AttemptNumber  int    `json:"attempt_number"`
	Kind           string `json:"kind"`
	ArtifactID     string `json:"artifact_id"`
	ChecksumSHA256 string `json:"checksum_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
	Output         string `json:"output"`
}

type MergeResult struct {
	JobID   string                 `json:"job_id"`
	State   string                 `json:"state"`
	Version int                    `json:"version"`
	Shards  []PublishedShardResult `json:"shards"`
}

func (p Policy) Normalized() Policy {
	out := p
	if out.MaxAttemptsPerShard <= 0 {
		out.MaxAttemptsPerShard = 2
	}
	if out.MaxActiveAttempts <= 0 {
		out.MaxActiveAttempts = 8
	}
	if out.LeaseMS <= 0 {
		out.LeaseMS = 5000
	}
	if out.SpeculateAfterMS <= 0 {
		out.SpeculateAfterMS = 1500
	}
	if out.HeartbeatIntervalMS <= 0 {
		out.HeartbeatIntervalMS = 500
	}
	return out
}
