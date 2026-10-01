package protocol

import "time"

// Estimator selects whether variance and covariance use Bessel correction.
type Estimator string

const (
	Population Estimator = "population"
	Sample     Estimator = "sample"
)

// JobRequest creates one distributed aggregation run.
type JobRequest struct {
	RunID          string    `json:"run_id"`
	Dimension      int       `json:"dimension"`
	Estimator      Estimator `json:"estimator"`
	ExpectedShards int       `json:"expected_shards"`
}

type Job struct {
	ID             string    `json:"id"`
	RunID          string    `json:"run_id"`
	Dimension      int       `json:"dimension"`
	Estimator      Estimator `json:"estimator"`
	ExpectedShards int       `json:"expected_shards"`
	Status         string    `json:"status"`
	ReceivedShards int       `json:"received_shards"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

// ShardRequest is one partition or worker contribution.
// Values is row-major: len(Values) == len(Weights)*Dimension.
type ShardRequest struct {
	ShardID    string    `json:"shard_id"`
	Values     []float64 `json:"values"`
	Weights    []float64 `json:"weights,omitempty"`
	ProducerID string    `json:"producer_id,omitempty"`
}

type ShardResult struct {
	JobID       string    `json:"job_id"`
	ShardID     string    `json:"shard_id"`
	Sequence    int64     `json:"sequence"`
	Observation int       `json:"observations"`
	Weight      float64   `json:"weight"`
	Summary     *Summary  `json:"summary"`
	ReceivedAt  time.Time `json:"received_at"`
}

type SummaryView struct {
	Count      int        `json:"count"`
	Weight     float64    `json:"weight"`
	Mean       []float64  `json:"mean"`
	Variance   []*float64 `json:"variance"`
	Covariance []*float64 `json:"covariance,omitempty"`
}

type JobResult struct {
	Job     Job          `json:"job"`
	Summary *SummaryView `json:"summary"`
}

type Event struct {
	Sequence   int64         `json:"sequence"`
	JobID      string        `json:"job_id"`
	RunID      string        `json:"run_id"`
	ShardID    string        `json:"shard_id,omitempty"`
	Type       string        `json:"type"`
	Reason     string        `json:"reason,omitempty"`
	ErrorClass string        `json:"error_class"`
	Detail     string        `json:"detail"`
	Before     *SummaryView  `json:"before,omitempty"`
	After      *SummaryView  `json:"after,omitempty"`
	Shard      *ShardRequest `json:"shard,omitempty"`
	CreatedAt  time.Time     `json:"created_at"`
}

type ErrorResponse struct {
	ErrorClass string `json:"error_class"`
	Code       string `json:"code"`
	Message    string `json:"message"`
	RunID      string `json:"run_id,omitempty"`
	JobID      string `json:"job_id,omitempty"`
	ShardID    string `json:"shard_id,omitempty"`
}

// Summary is the mergeable accumulator used by workers and the coordinator.
// Covariance is a dimension-by-dimension row-major symmetric matrix.
type Summary struct {
	Count      int       `json:"count"`
	Weight     float64   `json:"weight"`
	Mean       []float64 `json:"mean"`
	M2         []float64 `json:"m2"`
	Covariance []float64 `json:"covariance"`
}

func (s Summary) View(estimator Estimator) SummaryView {
	view := SummaryView{
		Count:  s.Count,
		Weight: s.Weight,
		Mean:   append([]float64(nil), s.Mean...),
	}
	if s.Weight <= 0 || len(s.Mean) == 0 {
		view.Mean = nil
		view.Variance = nil
		view.Covariance = nil
		return view
	}

	denominator := s.Weight
	if estimator == Sample {
		denominator = s.Weight - 1
	}
	view.Variance = make([]*float64, len(s.M2))
	for i, value := range s.M2 {
		if denominator > 0 {
			result := value / denominator
			view.Variance[i] = &result
		}
	}
	if len(s.Covariance) > 0 {
		view.Covariance = make([]*float64, len(s.Covariance))
		for i, value := range s.Covariance {
			if denominator > 0 {
				result := value / denominator
				view.Covariance[i] = &result
			}
		}
	}
	return view
}
