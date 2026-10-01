package compute

import "math"

// Metrics is the JSON-friendly result. Null covariance entries in the API are
// represented as nil slices when a denominator is undefined.
type Metrics struct {
	Dim                  int       `json:"dim"`
	N                    int64     `json:"n"`
	Weight               float64   `json:"weight"`
	Means                []float64 `json:"means"`
	PopulationCovariance []float64 `json:"population_covariance"`
	SampleCovariance     []float64 `json:"sample_covariance"`
}

func MetricsFromSummary(s *Summary) (*Metrics, error) {
	if err := ValidateSummary(s); err != nil {
		return nil, err
	}
	m := &Metrics{
		Dim:                  s.Dim,
		N:                    s.N,
		Weight:               s.Weight,
		Means:                append([]float64(nil), s.AbsoluteMeans()...),
		PopulationCovariance: nil,
		SampleCovariance:     nil,
	}
	if s.Weight > 0 {
		m.PopulationCovariance = make([]float64, s.Dim*s.Dim)
		for i, v := range s.M2 {
			m.PopulationCovariance[i] = (v + s.M2Comp[i]) / s.Weight
		}
	}
	if s.Weight > 1 {
		m.SampleCovariance = make([]float64, s.Dim*s.Dim)
		for i, v := range s.M2 {
			m.SampleCovariance[i] = (v + s.M2Comp[i]) / (s.Weight - 1)
		}
	}
	return m, nil
}

// SampleDefined reports whether a finite sample covariance denominator exists.
// This is explicit so empty shards and one-mass summaries do not look like
// ordinary zero results.
func SampleDefined(weight float64) bool {
	return weight > 1 && !math.IsNaN(weight)
}
