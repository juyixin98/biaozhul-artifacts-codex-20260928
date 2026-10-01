package stats

import (
	"math"
	"testing"

	"diststats/internal/protocol"
)

func TestFromRowsAndMergeAgainstExactRationalReference(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		values  []float64
		weights []float64
		dim     int
	}{
		{
			name:   "large offset tiny variance univariate",
			values: []float64{1e12, 1e12 + 1, 1e12 + 2, 1e12 + 3},
			dim:    1,
		},
		{
			name: "bivariate correlated around large offset",
			values: []float64{
				1e9, 1e9 + 4,
				1e9 + 1, 1e9 + 5,
				1e9 - 2, 1e9 + 2,
				1e9 + 3, 1e9 + 7,
			},
			dim: 2,
		},
		{
			name: "extremely uneven unweighted partitions",
			values: func() []float64 {
				values := []float64{1000000.25, 1000000.75}
				for i := 0; i < 500; i++ {
					values = append(values, 7.5+float64(i%5)*0.125)
				}
				return values
			}(),
			dim: 1,
		},
		{
			name: "uneven frequency weights",
			values: []float64{
				1e8 - 1, 5,
				1e8 + 1, 7,
				1e8 + 3, 9,
				1e8 - 7, 1,
			},
			weights: []float64{1e6, 1, 1e3, 1e9},
			dim:     2,
		},
		{
			name: "imperfectly correlated weighted bivariate",
			values: []float64{
				1, 2,
				2, -1,
				3, 7,
				5, 0,
			},
			weights: []float64{0.5, 3, 2, 8},
			dim:     2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			expected := exactReference(t, tc.values, tc.weights, tc.dim)
			rows := len(tc.values) / tc.dim

			t.Run("sequential", func(t *testing.T) {
				got, err := FromRows(tc.values, tc.weights, tc.dim)
				if err != nil {
					t.Fatalf("FromRows returned error: %v", err)
				}
				assertCloseSummary(t, got, expected, 4e-12)
				if tc.dim > 1 {
					for i := 0; i < tc.dim; i++ {
						for j := i + 1; j < tc.dim; j++ {
							if got.Covariance[i*tc.dim+j] != got.Covariance[j*tc.dim+i] {
								t.Fatalf("covariance matrix is not symmetric at (%d,%d)", i, j)
							}
						}
					}
				}
			})

			t.Run("left skewed merge tree", func(t *testing.T) {
				shards := buildShards(t, tc.values, tc.weights, tc.dim, []int{1, rows - 1})
				merged, err := Merge(shards[0], shards[1])
				if err != nil {
					t.Fatalf("Merge: %v", err)
				}
				assertCloseSummary(t, merged, expected, 4e-12)
			})

			t.Run("right skewed merge tree", func(t *testing.T) {
				shards := buildShards(t, tc.values, tc.weights, tc.dim, []int{rows - 1, 1})
				merged, err := Merge(shards[0], shards[1])
				if err != nil {
					t.Fatalf("Merge: %v", err)
				}
				assertCloseSummary(t, merged, expected, 4e-12)
			})

			t.Run("many tiny shards balanced tree", func(t *testing.T) {
				sizes := make([]int, rows)
				for i := range sizes {
					sizes[i] = 1
				}
				shards := buildShards(t, tc.values, tc.weights, tc.dim, sizes)
				for len(shards) > 1 {
					var next []*protocol.Summary
					for i := 0; i < len(shards); i += 2 {
						if i+1 == len(shards) {
							next = append(next, shards[i])
							break
						}
						merged, err := Merge(shards[i], shards[i+1])
						if err != nil {
							t.Fatalf("Merge(%d,%d): %v", i, i+1, err)
						}
						next = append(next, merged)
					}
					shards = next
				}
				assertCloseSummary(t, shards[0], expected, 2e-11)
			})
		})
	}
}

func TestEmptyZeroWeightAndSingleObservation(t *testing.T) {
	t.Parallel()

	empty, err := FromRows(nil, nil, 2)
	if err != nil {
		t.Fatalf("empty shard: %v", err)
	}
	if empty.Count != 0 || empty.Weight != 0 || len(empty.Mean) != 2 {
		t.Fatalf("unexpected empty summary: %+v", empty)
	}
	if view := empty.View(protocol.Population); view.Mean != nil || view.Variance != nil || view.Covariance != nil {
		t.Fatalf("empty shard statistics must be null, got %+v", view)
	}

	zeroWeight, err := FromRows([]float64{3, 9}, []float64{0}, 2)
	if err != nil {
		t.Fatalf("zero-weight row: %v", err)
	}
	if zeroWeight.Count != 1 || zeroWeight.Weight != 0 {
		t.Fatalf("zero-weight row should count as one observation with no mass: %+v", zeroWeight)
	}
	merged, err := Merge(zeroWeight, empty)
	if err != nil {
		t.Fatalf("merge zero-mass summaries: %v", err)
	}
	if merged.Count != 1 || merged.Weight != 0 {
		t.Fatalf("zero-mass merge changed mass: %+v", merged)
	}

	one, err := FromRows([]float64{4, -8}, nil, 2)
	if err != nil {
		t.Fatalf("single observation: %v", err)
	}
	pop := one.View(protocol.Population)
	sample := one.View(protocol.Sample)
	if pop.Mean[0] != 4 || pop.Mean[1] != -8 || *pop.Variance[0] != 0 || *pop.Covariance[3] != 0 {
		t.Fatalf("unexpected population view: %+v", pop)
	}
	if sample.Variance[0] != nil || sample.Covariance[0] != nil {
		t.Fatalf("sample variance with one effective observation must be null: %+v", sample)
	}
}

func TestRejectsInvalidInputAndIncompatibleDimensions(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		values    []float64
		weights   []float64
		dimension int
		code      string
	}{
		{"bad shape", []float64{1, 2, 3}, nil, 2, "INVALID_VALUES_SHAPE"},
		{"weights shape", []float64{1, 2}, []float64{1}, 1, "INVALID_WEIGHTS_SHAPE"},
		{"negative weight", []float64{1}, []float64{-0.1}, 1, "INVALID_WEIGHT"},
		{"non-finite weight", []float64{1}, []float64{math.NaN()}, 1, "INVALID_WEIGHT"},
		{"non-finite value", []float64{math.Inf(1)}, nil, 1, "INVALID_VALUE"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := FromRows(tc.values, tc.weights, tc.dimension)
			assertError(t, err, protocol.ClassInput, tc.code)
		})
	}

	a, _ := FromRows([]float64{1, 2}, nil, 2)
	b, _ := FromRows([]float64{1, 2, 3}, nil, 3)
	_, err := Merge(a, b)
	assertError(t, err, protocol.ClassInput, "DIMENSION_MISMATCH")
}

func buildShards(t *testing.T, values, weights []float64, dimension int, sizes []int) []*protocol.Summary {
	t.Helper()
	var shards []*protocol.Summary
	offsetRows := 0
	for _, size := range sizes {
		start := offsetRows * dimension
		end := (offsetRows + size) * dimension
		var shardWeights []float64
		if weights != nil {
			shardWeights = weights[offsetRows : offsetRows+size]
		}
		summary, err := FromRows(values[start:end], shardWeights, dimension)
		if err != nil {
			t.Fatalf("building shard at rows %d size %d: %v", offsetRows, size, err)
		}
		shards = append(shards, summary)
		offsetRows += size
	}
	if offsetRows != len(values)/dimension {
		t.Fatalf("shard sizes cover %d rows, want %d", offsetRows, len(values)/dimension)
	}
	return shards
}
