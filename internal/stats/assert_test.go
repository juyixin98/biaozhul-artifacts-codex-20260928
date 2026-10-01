package stats

import (
	"errors"
	"math"
	"testing"

	"diststats/internal/protocol"
)

func assertCloseSummary(t *testing.T, got *protocol.Summary, expected exactResult, tolerance float64) {
	t.Helper()
	if math.Abs(got.Weight-expected.weight) > 1e-12*math.Max(1, expected.weight) {
		t.Fatalf("weight = %.17g, want %.17g", got.Weight, expected.weight)
	}

	for i := range expected.mean {
		tol := tolerance * math.Max(1, math.Abs(expected.mean[i]))
		err := math.Abs(got.Mean[i] - expected.mean[i])
		t.Logf("mean[%d] got=%.17g want=%.17g abs_err=%.3e tol=%.3e",
			i, got.Mean[i], expected.mean[i], err, tol)
		if err > tol {
			t.Fatalf("mean[%d] error %.3e exceeds tolerance %.3e", i, err, tol)
		}
	}

	pop := got.View(protocol.Population)
	sample := got.View(protocol.Sample)
	for i, want := range expected.variance {
		tol := math.Max(1e-7, tolerance*math.Abs(want))
		gotValue := valueOrNaN(pop.Variance[i])
		err := math.Abs(gotValue - want)
		t.Logf("population variance[%d] got=%.17g want=%.17g abs_err=%.3e tol=%.3e",
			i, gotValue, want, err, tol)
		if err > tol {
			t.Fatalf("population variance[%d] error %.3e exceeds tolerance %.3e", i, err, tol)
		}

		if expected.weight <= 1 {
			if sample.Variance[i] != nil {
				t.Fatalf("sample variance[%d] must be undefined when weight %.17g <= 1", i, expected.weight)
			}
			continue
		}
		sampleWant := expected.variance[i] * expected.weight / (expected.weight - 1)
		sampleValue := valueOrNaN(sample.Variance[i])
		sampleErr := math.Abs(sampleValue - sampleWant)
		t.Logf("sample variance[%d] got=%.17g want=%.17g abs_err=%.3e tol=%.3e",
			i, sampleValue, sampleWant, sampleErr, tol)
		if sampleErr > tol {
			t.Fatalf("sample variance[%d] error %.3e exceeds tolerance %.3e", i, sampleErr, tol)
		}
	}

	for i, want := range expected.covariance {
		tol := math.Max(1e-7, tolerance*math.Abs(want))
		gotValue := got.Covariance[i] / expected.weight
		err := math.Abs(gotValue - want)
		t.Logf("population covariance[%d] got=%.17g want=%.17g abs_err=%.3e tol=%.3e",
			i, gotValue, want, err, tol)
		if err > tol {
			t.Fatalf("covariance[%d] error %.3e exceeds tolerance %.3e", i, err, tol)
		}
	}
}

func valueOrNaN(value *float64) float64 {
	if value == nil {
		return math.NaN()
	}
	return *value
}

func assertError(t *testing.T, err error, class protocol.ErrorClass, code string) {
	t.Helper()
	serviceErr, ok := protocol.AsServiceError(err)
	if !ok {
		t.Fatalf("error %v does not carry the service error contract", err)
	}
	if serviceErr.Class != class || serviceErr.Code != code {
		t.Fatalf("error = (%s,%s), want (%s,%s)", serviceErr.Class, serviceErr.Code, class, code)
	}
	if serviceErr.Message == "" {
		t.Fatal("service error must include a concrete message")
	}
	var target *protocol.ServiceError
	if !errors.As(err, &target) {
		t.Fatalf("errors.As cannot inspect service error: %v", err)
	}
}
