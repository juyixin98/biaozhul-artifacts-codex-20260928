package compute

import (
	"math"

	"diststats/internal/apperror"
)

const psdTolerance = 1e-12

func ValidateSummary(s *Summary) error {
	if s == nil {
		return apperror.InvalidInput("INVALID_SUMMARY", "summary is required")
	}
	if s.Dim <= 0 {
		return apperror.InvalidInput("INVALID_DIMENSION", "dimension must be positive")
	}
	if len(s.Mean) != s.Dim {
		return apperror.InvalidInput("INVALID_MEANS",
			"means length must equal dimension")
	}
	if s.MeanComp == nil {
		s.MeanComp = make([]float64, s.Dim)
	}
	if len(s.MeanComp) != s.Dim {
		return apperror.InvalidInput("INVALID_MEANS_COMP",
			"means compensation length must equal dimension")
	}
	if len(s.Origin) != s.Dim {
		return apperror.InvalidInput("INVALID_ORIGIN",
			"origin length must equal dimension")
	}
	if len(s.M2) != s.Dim*s.Dim {
		return apperror.InvalidInput("INVALID_M2",
			"m2 matrix length must equal dimension squared")
	}
	if s.M2Comp == nil {
		s.M2Comp = make([]float64, s.Dim*s.Dim)
	}
	if len(s.M2Comp) != s.Dim*s.Dim {
		return apperror.InvalidInput("INVALID_M2_COMP",
			"m2 compensation matrix length must equal dimension squared")
	}
	if s.S1 == nil {
		s.S1 = make([]float64, s.Dim)
	}
	if len(s.S1) != s.Dim {
		return apperror.InvalidInput("INVALID_S1", "s1 length must equal dimension")
	}
	if s.S1Comp == nil {
		s.S1Comp = make([]float64, s.Dim)
	}
	if len(s.S1Comp) != s.Dim {
		return apperror.InvalidInput("INVALID_S1_COMP",
			"s1 compensation length must equal dimension")
	}
	if math.IsNaN(s.Weight) || math.IsInf(s.Weight, 0) || s.Weight < 0 {
		return apperror.InvalidInput("INVALID_WEIGHT",
			"total weight must be finite and non-negative")
	}
	if s.N < 0 {
		return apperror.InvalidInput("INVALID_COUNT", "observation count must be non-negative")
	}
	if s.Weight == 0 && s.N != 0 {
		return apperror.InvalidInput("INVALID_SUMMARY",
			"zero-weight summary must have zero positive-weight observations")
	}
	if s.Weight > 0 && s.N == 0 {
		return apperror.InvalidInput("INVALID_SUMMARY",
			"positive-weight summary must have at least one positive-weight observation")
	}

	for i, v := range s.Mean {
		if math.IsNaN(s.Origin[i]) || math.IsInf(s.Origin[i], 0) {
			return apperror.InvalidInput("INVALID_ORIGIN",
				"all origin coordinates must be finite")
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return apperror.InvalidInput("INVALID_MEANS",
				"all means must be finite")
		}
		if c := s.MeanComp[i]; math.IsNaN(c) || math.IsInf(c, 0) {
			return apperror.InvalidInput("INVALID_MEANS_COMP",
				"all mean compensation values must be finite")
		}
		_ = i
	}
	for i := 0; i < s.Dim; i++ {
		for j := 0; j < s.Dim; j++ {
			v := s.M2[i*s.Dim+j]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return apperror.InvalidInput("INVALID_M2",
					"all m2 entries must be finite")
			}
			if c := s.M2Comp[i*s.Dim+j]; math.IsNaN(c) || math.IsInf(c, 0) {
				return apperror.InvalidInput("INVALID_M2_COMP",
					"all m2 compensation entries must be finite")
			}
			if c := s.S1Comp[i]; math.IsNaN(c) || math.IsInf(c, 0) {
				return apperror.InvalidInput("INVALID_S1_COMP",
					"all s1 compensation values must be finite")
			}
			if math.IsNaN(s.S1[i]) || math.IsInf(s.S1[i], 0) {
				return apperror.InvalidInput("INVALID_S1",
					"all s1 values must be finite")
			}
		}
		diag := s.M2[i*s.Dim+i]
		effectiveDiag := diag + s.M2Comp[i*s.Dim+i]
		if effectiveDiag < -psdTolerance*math.Max(1, math.Abs(effectiveDiag)) {
			return apperror.InvalidInput("INVALID_M2",
				"m2 matrix has negative diagonal variance")
		}
	}

	// Floating-point summaries submitted by external workers can be mildly
	// asymmetric. Symmetrize only inside that relative tolerance; reject a
	// structurally bad partition instead of silently hiding it.
	for i := 0; i < s.Dim; i++ {
		for j := i + 1; j < s.Dim; j++ {
			a := s.M2[i*s.Dim+j]
			b := s.M2[j*s.Dim+i]
			scale := math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
			if math.Abs(a-b) > psdTolerance*scale {
				return apperror.InvalidInput("INVALID_M2",
					"m2 covariance matrix is not symmetric")
			}
			mid := (a + b) / 2
			s.M2[i*s.Dim+j] = mid
			s.M2[j*s.Dim+i] = mid
			ca := s.M2Comp[i*s.Dim+j]
			cb := s.M2Comp[j*s.Dim+i]
			cscale := math.Max(1, math.Max(math.Abs(ca), math.Abs(cb)))
			if math.Abs(ca-cb) > psdTolerance*cscale {
				return apperror.InvalidInput("INVALID_M2_COMP",
					"m2 compensation matrix is not symmetric")
			}
			cmid := (ca + cb) / 2
			s.M2Comp[i*s.Dim+j] = cmid
			s.M2Comp[j*s.Dim+i] = cmid
		}
	}

	for i := 0; i < s.Dim; i++ {
		for j := i + 1; j < s.Dim; j++ {
			a := s.M2[i*s.Dim+i] + s.M2Comp[i*s.Dim+i]
			b := s.M2[j*s.Dim+j] + s.M2Comp[j*s.Dim+j]
			c := s.M2[i*s.Dim+j] + s.M2Comp[i*s.Dim+j]
			bound := math.Sqrt(math.Max(0, a) * math.Max(0, b))
			scale := math.Max(1, bound)
			if math.Abs(c) > bound+psdTolerance*scale {
				return apperror.InvalidInput("INVALID_M2",
					"m2 covariance matrix violates Cauchy-Schwarz")
			}
		}
	}
	return nil
}
