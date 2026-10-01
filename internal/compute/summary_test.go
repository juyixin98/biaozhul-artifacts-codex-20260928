package compute_test

import (
	"math"
	"math/big"
	"testing"

	"diststats/internal/apperror"
	"diststats/internal/compute"
)

type row struct {
	x      []float64
	weight float64
}

type refSummary struct {
	n      int64
	weight float64
	mean   []float64
	m2     []float64
}

func TestEmptyAndSingleSampleDefinitions(t *testing.T) {
	s := compute.Empty(2)
	m, err := compute.MetricsFromSummary(s)
	if err != nil {
		t.Fatal(err)
	}
	if m.Weight != 0 || m.N != 0 {
		t.Fatalf("empty summary = weight %v n %d, want zero identity", m.Weight, m.N)
	}
	if m.PopulationCovariance != nil || m.SampleCovariance != nil {
		t.Fatalf("empty covariance = pop:%v sample:%v, want nil for undefined denominators",
			m.PopulationCovariance, m.SampleCovariance)
	}

	if err := compute.AddPoint(s, []float64{2, -4}, 1); err != nil {
		t.Fatal(err)
	}
	m, err = compute.MetricsFromSummary(s)
	if err != nil {
		t.Fatal(err)
	}
	if m.PopulationCovariance[0] != 0 || m.PopulationCovariance[3] != 0 {
		t.Fatalf("population diagonal = %v, want zero", m.PopulationCovariance)
	}
	if m.SampleCovariance != nil {
		t.Fatalf("sample covariance with one positive-weight observation must be undefined, got %v",
			m.SampleCovariance)
	}

	zero := compute.Empty(1)
	if err := compute.AddPoint(zero, []float64{9}, 0); err != nil {
		t.Fatal(err)
	}
	if zero.N != 0 || zero.Weight != 0 {
		t.Fatalf("zero-weight point changed accumulator: n=%d weight=%v", zero.N, zero.Weight)
	}
}

func TestKnownWeightedMomentsAndCovariance(t *testing.T) {
	rows := []row{
		{[]float64{1, -1}, 2},
		{[]float64{3, 1}, 1},
		{[]float64{5, 2}, 0},
	}
	s, err := compute.FromObservations(2, values(rows), weights(rows))
	if err != nil {
		t.Fatal(err)
	}
	assertFloat(t, s.Weight, 3, 1e-14)
	assertFloat(t, s.Mean[0], 5.0/3.0, 1e-14)
	assertFloat(t, s.Mean[1], -1.0/3.0, 1e-14)
	m, err := compute.MetricsFromSummary(s)
	if err != nil {
		t.Fatal(err)
	}
	// M200 = 2*(1-5/3)^2 + 1*(3-5/3)^2 = 8/3.
	assertFloat(t, m.PopulationCovariance[0], 8.0/9.0, 1e-14)
	// Sample denominator is W-1 = 2.
	assertFloat(t, m.SampleCovariance[0], 4.0/3.0, 1e-14)
	// M201 = 2*(-2/3)*(-2/3) + 1*(4/3)*(4/3) = 8/3.
	assertFloat(t, m.PopulationCovariance[1], 8.0/9.0, 1e-14)
}

func TestLargeOffsetSmallVarianceMatchesHighPrecisionReference(t *testing.T) {
	rows := makeLargeOffsetRows(1001)
	partitions := makePartitions(rows, []int{1, 250, 750})
	assertTreesAgainstReference(t, rows, partitions, []float64{1e15, 1e15})
}

func TestExtremeUnevenPartitionsMatchHighPrecisionReference(t *testing.T) {
	rows := makeMixedRows(2001)
	partitions := makePartitions(rows, []int{1, 2, 1998})
	assertTreesAgainstReference(t, rows, partitions, nil)
}

func TestExtremeWeightsMatchHighPrecisionReference(t *testing.T) {
	rows := []row{
		{[]float64{-100, 7}, 1e-12},
		{[]float64{100, -8}, 1e-6},
		{[]float64{101, -8.5}, 1e6},
		{[]float64{-50, 10}, 0.25},
	}
	partitions := makePartitions(rows, []int{1, 1, 2})
	assertTreesAgainstReference(t, rows, partitions, nil)
}

func TestMergeTreesDoNotNeedBitwiseAssociativity(t *testing.T) {
	rows := makeLargeOffsetRows(501)
	partitions := makePartitions(rows, []int{1, 3, 7, 13, 477})
	sums := partitionSummaries(t, partitions, []float64{1e15, 1e15})

	left, err := compute.MergeTree(sums, false)
	if err != nil {
		t.Fatal(err)
	}
	balanced, err := compute.MergeTree(sums, true)
	if err != nil {
		t.Fatal(err)
	}
	meanErr := maxAbsDiff(left.AbsoluteMeans(), balanced.AbsoluteMeans())
	m2Err := maxAbsDiff(left.M2, balanced.M2)
	m2Tol := 1e-7 + 1e-10*maxAbsValue(left.M2)
	if meanErr > 1e-9 || m2Err > m2Tol {
		t.Fatalf("tree differences mean=%g m2=%g exceeded tolerance", meanErr, m2Err)
	}
	t.Logf("measured different-tree errors: mean=%g m2=%g", meanErr, m2Err)
}

func TestRejectsInvalidInputsAndIncompatibleDimensions(t *testing.T) {
	a := compute.Empty(2)
	err := compute.MergeInto(a, compute.Empty(3))
	assertErrorKind(t, err, apperror.KindInvalidInput, "DIMENSION_MISMATCH")

	if err := compute.AddPoint(compute.Empty(2), []float64{1, 2}, -1); err == nil {
		t.Fatal("negative weight was accepted")
	} else {
		assertErrorKind(t, err, apperror.KindInvalidInput, "INVALID_WEIGHT")
	}
	if err := compute.AddPoint(compute.Empty(2), []float64{1, math.NaN()}, 1); err == nil {
		t.Fatal("NaN observation was accepted")
	} else {
		assertErrorKind(t, err, apperror.KindInvalidInput, "INVALID_OBSERVATION")
	}

	bad := compute.Empty(1)
	bad.Weight = 1
	bad.N = 0
	err = compute.ValidateSummary(bad)
	assertErrorKind(t, err, apperror.KindInvalidInput, "INVALID_SUMMARY")
}

func assertTreesAgainstReference(t *testing.T, rows []row, partitions [][]row, origin []float64) {
	t.Helper()
	ref := highPrecisionReference(rows, 2)
	sums := partitionSummaries(t, partitions, origin)
	meanTol := scaledTolerance(1e-10, 1e-9, ref.mean)
	m2Tol := scaledTolerance(1e-10, 1e-8, ref.m2)
	for _, balanced := range []bool{false, true} {
		got, err := compute.MergeTree(sums, balanced)
		if err != nil {
			t.Fatal(err)
		}
		name := "sequential"
		if balanced {
			name = "balanced"
		}
		t.Run(name, func(t *testing.T) {
			if got.N != ref.n {
				t.Fatalf("n = %d, want %d", got.N, ref.n)
			}
			if math.Abs(got.Weight-ref.weight) > 1e-9*math.Max(1, ref.weight) {
				t.Fatalf("weight = %.17g, want %.17g", got.Weight, ref.weight)
			}
			meanErr := maxAbsDiff(got.AbsoluteMeans(), ref.mean)
			m2Err := maxAbsDiff(got.M2, ref.m2)
			t.Logf("measured errors against big.Float reference: mean=%g m2=%g; tolerances mean=%g m2=%g",
				meanErr, m2Err, meanTol, m2Tol)
			if meanErr > meanTol {
				t.Fatalf("max mean error %g exceeds tolerance %g", meanErr, meanTol)
			}
			if m2Err > m2Tol {
				t.Fatalf("max M2 error %g exceeds tolerance %g", m2Err, m2Tol)
			}
		})
	}
}

func partitionSummaries(t *testing.T, partitions [][]row, origin []float64) []*compute.Summary {
	t.Helper()
	out := make([]*compute.Summary, 0, len(partitions))
	for _, p := range partitions {
		s, err := compute.FromObservationsAt(2, origin, values(p), weights(p))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, s)
	}
	return out
}

func highPrecisionReference(rows []row, dim int) refSummary {
	const prec = 256
	wSum := new(big.Float).SetPrec(prec)
	for _, r := range rows {
		wSum.Add(wSum, new(big.Float).SetPrec(prec).SetFloat64(r.weight))
	}
	means := make([]*big.Float, dim)
	for i := range means {
		means[i] = new(big.Float).SetPrec(prec)
	}
	for _, r := range rows {
		w := new(big.Float).SetPrec(prec).SetFloat64(r.weight)
		for i, v := range r.x {
			term := new(big.Float).SetPrec(prec).SetFloat64(v)
			term.Mul(term, w)
			means[i].Add(means[i], term)
		}
	}
	for i := range means {
		means[i].Quo(means[i], wSum)
	}
	m2 := make([]*big.Float, dim*dim)
	for i := range m2 {
		m2[i] = new(big.Float).SetPrec(prec)
	}
	for _, r := range rows {
		w := new(big.Float).SetPrec(prec).SetFloat64(r.weight)
		dev := make([]*big.Float, dim)
		for i, v := range r.x {
			x := new(big.Float).SetPrec(prec).SetFloat64(v)
			dev[i] = new(big.Float).SetPrec(prec).Sub(x, means[i])
		}
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				term := new(big.Float).SetPrec(prec).Mul(dev[i], dev[j])
				term.Mul(term, w)
				m2[i*dim+j].Add(m2[i*dim+j], term)
			}
		}
	}
	out := refSummary{mean: make([]float64, dim), m2: make([]float64, dim*dim)}
	out.weight, _ = wSum.Float64()
	for _, r := range rows {
		if r.weight > 0 {
			out.n++
		}
	}
	for i := range out.mean {
		out.mean[i], _ = means[i].Float64()
	}
	for i := range out.m2 {
		out.m2[i], _ = m2[i].Float64()
	}
	return out
}

func makeLargeOffsetRows(n int) []row {
	out := make([]row, 0, n)
	for i := 0; i < n; i++ {
		base := 1e15
		dx := float64((i%5 - 2))
		dy := float64((i%7 - 3))
		if i%13 == 0 {
			dx += 0.25
		}
		out = append(out, row{[]float64{base + dx, base + dy}, 1})
	}
	return out
}

func makeMixedRows(n int) []row {
	out := make([]row, 0, n)
	for i := 0; i < n; i++ {
		w := 1.0
		if i%97 == 0 {
			w = 1000
		} else if i%31 == 0 {
			w = 0.001
		} else if i%23 == 0 {
			w = 0
		}
		base := 1e12
		if i%3 == 0 {
			base = -1e10
		}
		out = append(out, row{[]float64{
			base + float64(i%11)*0.125,
			float64(i%5) - 2 + float64(i%2)*0.5,
		}, w})
	}
	return out
}

func makePartitions(rows []row, sizes []int) [][]row {
	var out [][]row
	start := 0
	for _, size := range sizes {
		end := start + size
		out = append(out, rows[start:end])
		start = end
	}
	return out
}

func values(rows []row) [][]float64 {
	out := make([][]float64, len(rows))
	for i, r := range rows {
		out[i] = r.x
	}
	return out
}

func weights(rows []row) []float64 {
	out := make([]float64, len(rows))
	for i, r := range rows {
		out[i] = r.weight
	}
	return out
}

func assertFloat(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("got %.17g want %.17g", got, want)
	}
}

func assertErrorKind(t *testing.T, err error, kind apperror.Kind, code string) {
	t.Helper()
	ae, ok := err.(*apperror.Error)
	if !ok {
		t.Fatalf("error %v is not apperror.Error", err)
	}
	if ae.Kind != kind || ae.Code != code {
		t.Fatalf("error kind/code = %s/%s, want %s/%s", ae.Kind, ae.Code, kind, code)
	}
}

func maxAbsDiff(a, b []float64) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func maxAbsValue(a []float64) float64 {
	var m float64
	for _, v := range a {
		if math.Abs(v) > m {
			m = math.Abs(v)
		}
	}
	return m
}

func scaledTolerance(rel, floor float64, values []float64) float64 {
	tol := floor
	for _, v := range values {
		if candidate := floor + rel*math.Abs(v); candidate > tol {
			tol = candidate
		}
	}
	return tol
}
