package participant

import "opp300-speculative-batch/internal/protocol"

type ExecuteRequest struct {
	AttemptID           string            `json:"attempt_id"`
	JobID               string            `json:"job_id"`
	ShardID             string            `json:"shard_id"`
	Ordinal             int               `json:"ordinal"`
	Key                 string            `json:"key"`
	AttemptNumber       int               `json:"attempt_number"`
	Kind                string            `json:"kind"`
	Epoch               string            `json:"epoch"`
	Input               string            `json:"input"`
	Behavior            protocol.Behavior `json:"behavior"`
	CallbackBaseURL     string            `json:"callback_base_url"`
	LeaseMS             int               `json:"lease_ms"`
	HeartbeatIntervalMS int               `json:"heartbeat_interval_ms"`
}

type StartRequest struct {
	Epoch     string `json:"epoch"`
	WorkerURL string `json:"worker_url"`
}

type FailureRequest struct {
	Epoch           string `json:"epoch"`
	FailureCategory string `json:"failure_category"`
	FailureMessage  string `json:"failure_message"`
	WorkerURL       string `json:"worker_url"`
}

type StageResponse struct {
	StagingID      string `json:"staging_id"`
	ChecksumSHA256 string `json:"checksum_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
}

type CompleteRequest struct {
	Epoch          string `json:"epoch"`
	StagingID      string `json:"staging_id"`
	ChecksumSHA256 string `json:"checksum_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
}

type ErrorResponse struct {
	Error           string `json:"error"`
	FailureCategory string `json:"failure_category,omitempty"`
}
