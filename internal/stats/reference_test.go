package stats

import (
	"math/big"
	"testing"
)

type exactResult struct {
	weight     float64
	mean       []float64
	variance   []float64
	covariance []float64
}

func exactReference(t *testing.T, values, weights []float64, dimension int) exactResult {
	t.Helper()
	rows := len(values) / dimension
	weightRats := make([]*big.Rat, rows)
	weightSum := new(big.Rat)

	for row := 0; row < rows; row++ {
		weight := 1.0
		if weights != nil {
			weight = weights[row]
		}
		weightRats[row] = new(big.Rat).SetFloat64(weight)
		if weightRats[row] == nil {
			t.Fatalf("test weight %.17g cannot be represented as an exact rational float64", weight)
		}
		weightSum.Add(weightSum, weightRats[row])
	}

	mean := make([]*big.Rat, dimension)
	for i := range mean {
		mean[i] = new(big.Rat)
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < dimension; col++ {
			value := new(big.Rat).SetFloat64(values[row*dimension+col])
			mean[col].Add(mean[col], new(big.Rat).Mul(weightRats[row], value))
		}
	}
	for col := range mean {
		if weightSum.Sign() == 0 {
			mean[col] = nil
			continue
		}
		mean[col].Quo(mean[col], weightSum)
	}

	variance := make([]*big.Rat, dimension)
	covariance := make([]*big.Rat, dimension*dimension)
	for i := range variance {
		variance[i] = new(big.Rat)
	}
	for i := range covariance {
		covariance[i] = new(big.Rat)
	}
	for row := 0; row < rows; row++ {
		for i := 0; i < dimension; i++ {
			if mean[i] == nil {
				continue
			}
			di := new(big.Rat).Sub(new(big.Rat).SetFloat64(values[row*dimension+i]), mean[i])
			variance[i].Add(variance[i], new(big.Rat).Mul(weightRats[row], new(big.Rat).Mul(di, di)))
			for j := 0; j < dimension; j++ {
				dj := new(big.Rat).Sub(new(big.Rat).SetFloat64(values[row*dimension+j]), mean[j])
				covariance[i*dimension+j].Add(covariance[i*dimension+j],
					new(big.Rat).Mul(weightRats[row], new(big.Rat).Mul(di, dj)))
			}
		}
	}

	result := exactResult{
		weight: ratFloat(weightSum),
	}
	for _, value := range mean {
		result.mean = append(result.mean, ratFloat(value))
	}
	for _, value := range variance {
		result.variance = append(result.variance, ratFloat(new(big.Rat).Quo(value, weightSum)))
	}
	for _, value := range covariance {
		result.covariance = append(result.covariance, ratFloat(new(big.Rat).Quo(value, weightSum)))
	}
	return result
}

func ratFloat(value *big.Rat) float64 {
	if value == nil {
		return 0
	}
	result, _ := new(big.Float).SetRat(value).Float64()
	return result
}
