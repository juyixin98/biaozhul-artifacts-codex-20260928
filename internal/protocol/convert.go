package protocol

import (
	"encoding/json"

	"diststats/internal/compute"
)

func ToSummaryDTO(s *compute.Summary) SummaryDTO {
	return SummaryDTO{
		Dimension: s.Dim,
		N:         s.N,
		Weight:    s.Weight,
		Origin:    append([]float64(nil), s.Origin...),
		Means:     append([]float64(nil), s.Mean...),
		MeansComp: append([]float64(nil), s.MeanComp...),
		S1:        append([]float64(nil), s.S1...),
		S1Comp:    append([]float64(nil), s.S1Comp...),
		M2:        append([]float64(nil), s.M2...),
		M2Comp:    append([]float64(nil), s.M2Comp...),
	}
}

func FromSummaryDTO(d SummaryDTO) *compute.Summary {
	origin := d.Origin
	if len(origin) == 0 {
		origin = make([]float64, d.Dimension)
	}
	return &compute.Summary{
		Dim:      d.Dimension,
		N:        d.N,
		Weight:   d.Weight,
		Origin:   append([]float64(nil), origin...),
		Mean:     append([]float64(nil), d.Means...),
		MeanComp: append([]float64(nil), d.MeansComp...),
		S1:       append([]float64(nil), d.S1...),
		S1Comp:   append([]float64(nil), d.S1Comp...),
		M2:       append([]float64(nil), d.M2...),
		M2Comp:   append([]float64(nil), d.M2Comp...),
	}
}

func ToMetricsDTO(m *compute.Metrics) MetricsDTO {
	return MetricsDTO{
		Dimension:            m.Dim,
		N:                    m.N,
		Weight:               m.Weight,
		Means:                append([]float64(nil), m.Means...),
		PopulationCovariance: appendOrNil(m.PopulationCovariance),
		SampleCovariance:     appendOrNil(m.SampleCovariance),
	}
}

func appendOrNil(v []float64) []float64 {
	if v == nil {
		return nil
	}
	return append([]float64(nil), v...)
}

func RawJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}
