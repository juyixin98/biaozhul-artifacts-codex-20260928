package compute

import (
	"math"

	"diststats/internal/apperror"
)

// Summary is a weighted, mergeable multivariate moment accumulator.
//
// Mean and M2 are stored relative to Origin. Workers should choose an origin
// close to their local values so large common offsets do not make small
// residuals vanish. Merging uses absolute mean differences, not squared sums.
type Summary struct {
	Dim      int       `json:"dim"`
	N        int64     `json:"n"`
	Weight   float64   `json:"weight"`
	Origin   []float64 `json:"origin"`
	Mean     []float64 `json:"means"`
	MeanComp []float64 `json:"means_comp"`
	S1       []float64 `json:"s1"`
	S1Comp   []float64 `json:"s1_comp"`
	M2       []float64 `json:"m2"`
	M2Comp   []float64 `json:"m2_comp"`
}

func Empty(dim int) *Summary { return EmptyAt(dim, nil) }

func EmptyAt(dim int, origin []float64) *Summary {
	if dim <= 0 {
		return &Summary{Origin: []float64{}, Mean: []float64{}, M2: []float64{}}
	}
	o := make([]float64, dim)
	copy(o, origin)
	return &Summary{
		Dim:      dim,
		Origin:   o,
		Mean:     make([]float64, dim),
		MeanComp: make([]float64, dim),
		S1:       make([]float64, dim),
		S1Comp:   make([]float64, dim),
		M2:       make([]float64, dim*dim),
		M2Comp:   make([]float64, dim*dim),
	}
}

func Clone(s *Summary) *Summary {
	if s == nil {
		return nil
	}
	cp := &Summary{Dim: s.Dim, N: s.N, Weight: s.Weight}
	if s.Origin != nil {
		cp.Origin = append([]float64(nil), s.Origin...)
	}
	if s.Mean != nil {
		cp.Mean = append([]float64(nil), s.Mean...)
	}
	if s.MeanComp != nil {
		cp.MeanComp = append([]float64(nil), s.MeanComp...)
	}
	if s.S1 != nil {
		cp.S1 = append([]float64(nil), s.S1...)
	}
	if s.S1Comp != nil {
		cp.S1Comp = append([]float64(nil), s.S1Comp...)
	}
	if s.M2 != nil {
		cp.M2 = append([]float64(nil), s.M2...)
	}
	if s.M2Comp != nil {
		cp.M2Comp = append([]float64(nil), s.M2Comp...)
	}
	return cp
}

func AddPoint(s *Summary, x []float64, weight float64) error {
	if err := ValidateSummary(s); err != nil {
		return err
	}
	if len(x) != s.Dim {
		return apperror.InvalidInput("DIMENSION_MISMATCH",
			"observation dimension does not match accumulator dimension")
	}
	if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 {
		return apperror.InvalidInput("INVALID_WEIGHT",
			"weight must be a finite non-negative number")
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return apperror.InvalidInput("INVALID_OBSERVATION",
				"observations must be finite numbers")
		}
	}
	if weight == 0 {
		return nil
	}

	oldWeight := s.Weight
	newWeight := oldWeight + weight
	if math.IsInf(newWeight, 0) {
		return apperror.ComputationFailed("WEIGHT_OVERFLOW",
			"total weight exceeds the finite float64 range")
	}
	scale := weight / newWeight
	oldScale := oldWeight / newWeight

	for i := 0; i < s.Dim; i++ {
		deltaI := (x[i] - s.Origin[i]) - s.Mean[i]
		addCompensated(s.S1, s.S1Comp, i, weight*(x[i]-s.Origin[i]))
		for j := i; j < s.Dim; j++ {
			deltaJ := (x[j] - s.Origin[j]) - s.Mean[j]
			addCompensated(s.M2, s.M2Comp, i*s.Dim+j, weight*oldScale*deltaI*deltaJ)
			s.M2[j*s.Dim+i] = s.M2[i*s.Dim+j]
			s.M2Comp[j*s.Dim+i] = s.M2Comp[i*s.Dim+j]
		}
		s.Mean[i] += scale * deltaI
	}

	s.Weight = newWeight
	s.N++
	if err := ValidateSummary(s); err != nil {
		return apperror.Wrap(apperror.KindComputationFailed, "MOMENT_OVERFLOW",
			"moment update produced an invalid finite accumulator", err)
	}
	return nil
}

func FromObservations(dim int, rows [][]float64, weights []float64) (*Summary, error) {
	return FromObservationsAt(dim, nil, rows, weights)
}

func FromObservationsAt(dim int, origin []float64, rows [][]float64, weights []float64) (*Summary, error) {
	if dim <= 0 {
		return nil, apperror.InvalidInput("INVALID_DIMENSION", "dimension must be positive")
	}
	if weights != nil && len(weights) != len(rows) {
		return nil, apperror.InvalidInput("INVALID_WEIGHT_COUNT",
			"weights length must equal observations length")
	}
	if len(origin) > 0 && len(origin) != dim {
		return nil, apperror.InvalidInput("INVALID_ORIGIN", "origin dimension is invalid")
	}
	for _, v := range origin {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, apperror.InvalidInput("INVALID_ORIGIN", "origin must be finite")
		}
	}
	s := EmptyAt(dim, origin)
	weightedResidual := make([]float64, dim)
	weightedResidualComp := make([]float64, dim)
	var weightSum float64
	var weightComp float64
	for i, row := range rows {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if len(row) != dim {
			return nil, apperror.InvalidInput("DIMENSION_MISMATCH",
				"observation dimension does not match accumulator dimension")
		}
		for _, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, apperror.InvalidInput("INVALID_OBSERVATION",
					"observations must be finite numbers")
			}
		}
		if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
			return nil, apperror.InvalidInput("INVALID_WEIGHT",
				"weight must be a finite non-negative number")
		}
		if w == 0 {
			continue
		}
		weightSum, weightComp = addScalarCompensated(weightSum, weightComp, w)
		if math.IsInf(weightSum, 0) {
			return nil, apperror.ComputationFailed("WEIGHT_OVERFLOW",
				"total weight exceeds the finite float64 range")
		}
		s.N++
		for k, v := range row {
			addCompensated(weightedResidual, weightedResidualComp, k, w*(v-s.Origin[k]))
		}
	}
	if weightSum > 0 {
		weightSum += weightComp
		exactMean := make([]float64, dim)
		for k := range weightedResidual {
			rounded := weightedResidual[k] / weightSum
			exact := (weightedResidual[k] + weightedResidualComp[k]) / weightSum
			s.Mean[k] = rounded
			s.MeanComp[k] = exact - rounded
			exactMean[k] = exact
			s.S1[k] = weightedResidual[k]
			s.S1Comp[k] = weightedResidualComp[k]
		}
		for i, row := range rows {
			w := 1.0
			if weights != nil {
				w = weights[i]
			}
			if w == 0 {
				continue
			}
			for k := 0; k < dim; k++ {
				dk := (row[k] - s.Origin[k]) - exactMean[k]
				for l := k; l < dim; l++ {
					dl := (row[l] - s.Origin[l]) - exactMean[l]
					idx := k*dim + l
					addCompensated(s.M2, s.M2Comp, idx, w*dk*dl)
					s.M2[l*dim+k] = s.M2[idx]
					s.M2Comp[l*dim+k] = s.M2Comp[idx]
				}
			}
		}
		s.Weight = weightSum
		FoldCompensation(s)
	}
	if err := ValidateSummary(s); err != nil {
		return nil, err
	}
	return s, nil
}

func addScalarCompensated(sum, comp, x float64) (float64, float64) {
	t := sum + x
	if math.Abs(sum) >= math.Abs(x) {
		comp += (sum - t) + x
	} else {
		comp += (x - t) + sum
	}
	return t, comp
}

func Merge(a, b *Summary) (*Summary, error) {
	out := Clone(a)
	if out == nil {
		out = Empty(b.Dim)
	}
	if err := MergeInto(out, b); err != nil {
		return nil, err
	}
	return out, nil
}

func MergeInto(dst, src *Summary) error {
	if err := ValidateSummary(dst); err != nil {
		return err
	}
	if err := ValidateSummary(src); err != nil {
		return err
	}
	if dst.Dim != src.Dim {
		return apperror.InvalidInput("DIMENSION_MISMATCH",
			"cannot merge summaries with different dimensions")
	}
	if src.Weight == 0 {
		return nil
	}
	if dst.Weight == 0 {
		dst.N = src.N
		dst.Weight = src.Weight
		copy(dst.Origin, src.Origin)
		copy(dst.Mean, src.Mean)
		copy(dst.MeanComp, src.MeanComp)
		copy(dst.S1, src.S1)
		copy(dst.S1Comp, src.S1Comp)
		copy(dst.M2, src.M2)
		copy(dst.M2Comp, src.M2Comp)
		return ValidateSummary(dst)
	}

	total := dst.Weight + src.Weight
	if math.IsInf(total, 0) {
		return apperror.ComputationFailed("WEIGHT_OVERFLOW",
			"merged total weight exceeds the finite float64 range")
	}
	srcMeanForDelta := make([]float64, dst.Dim)
	dstMeanForDelta := make([]float64, dst.Dim)
	if SameOrigin(dst, src) {
		for i := 0; i < dst.Dim; i++ {
			srcMeanForDelta[i] = (src.S1[i] + src.S1Comp[i]) / src.Weight
			dstMeanForDelta[i] = (dst.S1[i] + dst.S1Comp[i]) / dst.Weight
		}
	} else {
		srcAbs := src.absoluteMeans()
		dstAbs := dst.absoluteMeans()
		copy(srcMeanForDelta, srcAbs)
		copy(dstMeanForDelta, dstAbs)
	}
	deltaFactor := dst.Weight * src.Weight / total
	newWeightedMean := make([]float64, dst.Dim)
	newWeightedMeanComp := make([]float64, dst.Dim)
	for i := 0; i < dst.Dim; i++ {
		if SameOrigin(dst, src) {
			addCompensated(newWeightedMean, newWeightedMeanComp, i, dst.S1[i])
			addCompensated(newWeightedMean, newWeightedMeanComp, i, dst.S1Comp[i])
			addCompensated(newWeightedMean, newWeightedMeanComp, i, src.S1[i])
			addCompensated(newWeightedMean, newWeightedMeanComp, i, src.S1Comp[i])
		} else {
			addCompensated(newWeightedMean, newWeightedMeanComp, i,
				dst.Weight*dst.Origin[i]+dst.S1[i]+dst.S1Comp[i])
			addCompensated(newWeightedMean, newWeightedMeanComp, i,
				src.Weight*src.Origin[i]+src.S1[i]+src.S1Comp[i])
		}
	}

	for i := 0; i < dst.Dim; i++ {
		deltaI := srcMeanForDelta[i] - dstMeanForDelta[i]
		for j := i; j < dst.Dim; j++ {
			deltaJ := srcMeanForDelta[j] - dstMeanForDelta[j]
			idx := i*dst.Dim + j
			addCompensated(dst.M2, dst.M2Comp, idx, src.M2[idx])
			addCompensated(dst.M2, dst.M2Comp, idx, src.M2Comp[idx])
			addCompensated(dst.M2, dst.M2Comp, idx, deltaFactor*deltaI*deltaJ)
			dst.M2[j*dst.Dim+i] = dst.M2[idx]
			dst.M2Comp[j*dst.Dim+i] = dst.M2Comp[idx]
		}
		meanRounded := newWeightedMean[i] / total
		meanExact := (newWeightedMean[i] + newWeightedMeanComp[i]) / total
		dst.Mean[i] = meanRounded
		if !SameOrigin(dst, src) {
			dst.Mean[i] = meanRounded - dst.Origin[i]
		}
		dst.MeanComp[i] = meanExact - meanRounded
		dst.S1[i] = newWeightedMean[i] - total*dst.Origin[i]
		dst.S1Comp[i] = newWeightedMeanComp[i]
		dst.S1[i] = newWeightedMean[i]
		dst.S1Comp[i] = newWeightedMeanComp[i]
	}
	dst.Weight = total
	dst.N += src.N
	FoldCompensation(dst)

	if err := ValidateSummary(dst); err != nil {
		return apperror.Wrap(apperror.KindComputationFailed, "MOMENT_OVERFLOW",
			"merge produced an invalid finite accumulator", err)
	}
	return nil
}

func FoldCompensation(s *Summary) {
	for i := range s.M2 {
		s.M2[i] += s.M2Comp[i]
		s.M2Comp[i] = 0
	}
}

func addCompensated(sum, comp []float64, idx int, x float64) {
	t := sum[idx] + x
	if math.Abs(sum[idx]) >= math.Abs(x) {
		comp[idx] += (sum[idx] - t) + x
	} else {
		comp[idx] += (x - t) + sum[idx]
	}
	sum[idx] = t
}

func (s *Summary) absoluteMeans() []float64 {
	out := make([]float64, s.Dim)
	for i := range out {
		out[i] = s.Origin[i] + (s.S1[i]+s.S1Comp[i])/s.Weight
	}
	return out
}

func (s *Summary) AbsoluteMeans() []float64 { return s.absoluteMeans() }

func SameOrigin(a, b *Summary) bool {
	for i := range a.Origin {
		if a.Origin[i] != b.Origin[i] {
			return false
		}
	}
	return true
}
