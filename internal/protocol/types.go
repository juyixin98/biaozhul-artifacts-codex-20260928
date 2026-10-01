package protocol

import "time"

type CreateJobRequest struct {
	Dimension int `json:"dimension"`
}

type JobResponse struct {
	ID          string     `json:"id"`
	Dimension   int        `json:"dimension"`
	Status      string     `json:"status"`
	Version     int64      `json:"version"`
	CreatedAt   time.Time  `json:"created_at"`
	FinalizedAt *time.Time `json:"finalized_at,omitempty"`
	Summary     SummaryDTO `json:"summary"`
}

type ObservationDTO struct {
	Values []float64 `json:"values"`
	Weight *float64  `json:"weight,omitempty"`
}

type SummaryDTO struct {
	Dimension int       `json:"dim"`
	N         int64     `json:"n"`
	Weight    float64   `json:"weight"`
	Origin    []float64 `json:"origin,omitempty"`
	Means     []float64 `json:"means"`
	MeansComp []float64 `json:"means_comp,omitempty"`
	S1        []float64 `json:"s1,omitempty"`
	S1Comp    []float64 `json:"s1_comp,omitempty"`
	M2        []float64 `json:"m2"`
	M2Comp    []float64 `json:"m2_comp,omitempty"`
}

type SubmitShardRequest struct {
	ShardID      string           `json:"shard_id"`
	Origin       []float64        `json:"origin,omitempty"`
	Observations []ObservationDTO `json:"observations,omitempty"`
	Summary      *SummaryDTO      `json:"summary,omitempty"`
}

type ShardResponse struct {
	JobID      string     `json:"job_id"`
	ShardID    string     `json:"shard_id"`
	Seq        int64      `json:"seq"`
	RunID      string     `json:"run_id"`
	RunNumber  int64      `json:"run_number"`
	CreatedAt  time.Time  `json:"created_at"`
	JobVersion int64      `json:"job_version"`
	Summary    SummaryDTO `json:"summary"`
	JobSummary SummaryDTO `json:"job_summary"`
}

type MetricsDTO struct {
	Dimension            int       `json:"dim"`
	N                    int64     `json:"n"`
	Weight               float64   `json:"weight"`
	Means                []float64 `json:"means"`
	PopulationCovariance []float64 `json:"population_covariance"`
	SampleCovariance     []float64 `json:"sample_covariance"`
}

type ResultResponse struct {
	JobID       string      `json:"job_id"`
	Finalized   bool        `json:"finalized"`
	ShardCount  int         `json:"shard_count"`
	ReplayOrder string      `json:"replay_order"`
	Summary     SummaryDTO  `json:"summary"`
	Metrics     *MetricsDTO `json:"metrics,omitempty"`
}

type RunEventResponse struct {
	Index     int64     `json:"index"`
	Stage     string    `json:"stage"`
	Result    string    `json:"result"`
	Reason    string    `json:"reason"`
	State     any       `json:"state"`
	CreatedAt time.Time `json:"created_at"`
}

type RunResponse struct {
	ID         string             `json:"id"`
	Number     int64              `json:"number"`
	JobID      string             `json:"job_id,omitempty"`
	Route      string             `json:"route"`
	Method     string             `json:"method"`
	StatusCode int                `json:"status_code,omitempty"`
	Status     string             `json:"status"`
	Request    any                `json:"request,omitempty"`
	Error      *ErrorResponse     `json:"error,omitempty"`
	StartedAt  time.Time          `json:"started_at"`
	FinishedAt *time.Time         `json:"finished_at,omitempty"`
	Events     []RunEventResponse `json:"events"`
}

type ErrorResponse struct {
	Kind    string `json:"kind"`
	Code    string `json:"code"`
	Message string `json:"message"`
}
