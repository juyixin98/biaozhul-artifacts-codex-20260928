package stats

import (
	"fmt"
	"math"

	"diststats/internal/protocol"
)

// NewSummary returns an explicitly dimensioned empty accumulator.
func NewSummary(dimension int) (*protocol.Summary, error) {
	if dimension <= 0 {
		return nil, protocol.NewServiceError(protocol.ClassInput, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be positive, got %d", dimension), nil)
	}
	return &protocol.Summary{
		Mean:       make([]float64, dimension),
		M2:         make([]float64, dimension),
		Covariance: make([]float64, dimension*dimension),
	}, nil
}

// FromRows builds a weighted Welford accumulator. Values is row-major.
// Weights are non-negative finite frequency weights: w>0 counts as one
// observation; w=0 is permitted and contributes no mass.
func FromRows(values []float64, weights []float64, dimension int) (*protocol.Summary, error) {
	if dimension <= 0 {
		return nil, protocol.NewServiceError(protocol.ClassInput, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be positive, got %d", dimension), nil)
	}
	if len(values)%dimension != 0 {
		return nil, protocol.NewServiceError(protocol.ClassInput, "INVALID_VALUES_SHAPE",
			fmt.Sprintf("len(values)=%d is not divisible by dimension=%d", len(values), dimension), nil)
	}
	rows := len(values) / dimension
	if len(weights) != 0 && len(weights) != rows {
		return nil, protocol.NewServiceError(protocol.ClassInput, "INVALID_WEIGHTS_SHAPE",
			fmt.Sprintf("got %d weights for %d observations", len(weights), rows), nil)
	}

	summary, err := NewSummary(dimension)
	if err != nil {
		return nil, err
	}

	for row := 0; row < rows; row++ {
		weight := 1.0
		if len(weights) == rows {
			weight = weights[row]
		}
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 {
			return nil, protocol.NewServiceError(protocol.ClassInput, "INVALID_WEIGHT",
				fmt.Sprintf("weight at row %d must be a finite, non-negative number", row), nil)
		}

		base := row * dimension
		for i := 0; i < dimension; i++ {
			value := values[base+i]
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, protocol.NewServiceError(protocol.ClassInput, "INVALID_VALUE",
					fmt.Sprintf("value at row %d, dimension %d must be finite", row, i), nil)
			}
		}

		if weight == 0 {
			summary.Count++
			continue
		}
		if err := addWeightedRow(summary, values[base:base+dimension], weight); err != nil {
			return nil, err
		}
		summary.Count++
	}
	return summary, nil
}

func addWeightedRow(summary *protocol.Summary, row []float64, weight float64) error {
	dimension := len(summary.Mean)
	newWeight := summary.Weight + weight
	if math.IsInf(newWeight, 0) || math.IsNaN(newWeight) {
		return protocol.NewServiceError(protocol.ClassComputation, "WEIGHT_OVERFLOW",
			"aggregate weight exceeds the finite float64 range", nil)
	}
	scale := weight / newWeight
	mergeFactor := summary.Weight * weight / newWeight
	oldMean := append([]float64(nil), summary.Mean...)
	oldDelta := make([]float64, dimension)

	for i := 0; i < dimension; i++ {
		oldDelta[i] = row[i] - oldMean[i]
		summary.Mean[i] = oldMean[i] + scale*oldDelta[i]
		summary.M2[i] += mergeFactor * oldDelta[i] * oldDelta[i]
	}

	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			idx := i*dimension + j
			summary.Covariance[idx] += mergeFactor * oldDelta[i] * oldDelta[j]
		}
	}
	summary.Weight = newWeight
	if hasNonFinite(summary.Mean) || hasNonFinite(summary.M2) || hasNonFinite(summary.Covariance) {
		return protocol.NewServiceError(protocol.ClassComputation, "NON_FINITE_STATISTIC",
			"a mean, variance accumulator, or covariance accumulator became non-finite", nil)
	}
	return nil
}

// Merge combines two independently computed summaries using Chan's parallel
// variance formula and its multivariate covariance extension. It never forms
// a difference of two large sums of squares.
func Merge(a, b *protocol.Summary) (*protocol.Summary, error) {
	if a == nil || b == nil {
		return nil, protocol.NewServiceError(protocol.ClassInput, "MISSING_SUMMARY",
			"both summaries are required", nil)
	}
	dimension := len(a.Mean)
	if dimension == 0 || len(b.Mean) == 0 {
		return nil, protocol.NewServiceError(protocol.ClassInput, "INVALID_DIMENSION",
			"summaries must have an explicit positive dimension", nil)
	}
	if len(b.Mean) != dimension || len(a.M2) != dimension || len(b.M2) != dimension ||
		len(a.Covariance) != dimension*dimension || len(b.Covariance) != dimension*dimension {
		return nil, protocol.NewServiceError(protocol.ClassInput, "DIMENSION_MISMATCH",
			fmt.Sprintf("cannot merge dimensions %d and %d", dimension, len(b.Mean)), nil)
	}
	if err := validateSummary(a); err != nil {
		return nil, err
	}
	if err := validateSummary(b); err != nil {
		return nil, err
	}

	out, _ := NewSummary(dimension)
	totalWeight := a.Weight + b.Weight
	if math.IsInf(totalWeight, 0) || math.IsNaN(totalWeight) {
		return nil, protocol.NewServiceError(protocol.ClassComputation, "WEIGHT_OVERFLOW",
			"merged weight exceeds the finite float64 range", nil)
	}
	out.Count = a.Count + b.Count
	out.Weight = totalWeight
	if totalWeight == 0 {
		return out, nil
	}

	factor := a.Weight * b.Weight / totalWeight
	for i := 0; i < dimension; i++ {
		delta := b.Mean[i] - a.Mean[i]
		out.Mean[i] = a.Mean[i] + delta*(b.Weight/totalWeight)
		out.M2[i] = a.M2[i] + b.M2[i] + delta*delta*factor
	}
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			idx := i*dimension + j
			deltaI := b.Mean[i] - a.Mean[i]
			deltaJ := b.Mean[j] - a.Mean[j]
			out.Covariance[idx] = a.Covariance[idx] + b.Covariance[idx] + deltaI*deltaJ*factor
		}
	}
	if hasNonFinite(out.Mean) || hasNonFinite(out.M2) || hasNonFinite(out.Covariance) {
		return nil, protocol.NewServiceError(protocol.ClassComputation, "NON_FINITE_STATISTIC",
			"merged statistic became non-finite", nil)
	}
	return out, nil
}

func validateSummary(summary *protocol.Summary) error {
	if summary.Weight < 0 || math.IsNaN(summary.Weight) || math.IsInf(summary.Weight, 0) {
		return protocol.NewServiceError(protocol.ClassInput, "INVALID_WEIGHT",
			"summary weight must be finite and non-negative", nil)
	}
	if summary.Count < 0 {
		return protocol.NewServiceError(protocol.ClassInput, "INVALID_COUNT", "summary count cannot be negative", nil)
	}
	if hasNonFinite(summary.Mean) || hasNonFinite(summary.M2) || hasNonFinite(summary.Covariance) {
		return protocol.NewServiceError(protocol.ClassInput, "NON_FINITE_STATISTIC",
			"summary contains non-finite values", nil)
	}
	return nil
}

func hasNonFinite(values []float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return true
		}
	}
	return false
}
