"""Hand-computed fixtures assert exact weighted calibration statistics."""

from __future__ import annotations

import math

import pytest

from app.kernel import equal_frequency_bins, fixed_bins
from tests.reference_stats import fixed_diagnostics, quantile_diagnostics


def assert_close(actual, expected, tolerance=1e-12):
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, abs=tolerance)


def test_perfect_calibration_fixed_bins_has_zero_reliability(run_identity):
    p = [0.0, 0.1, 0.9, 1.0]
    y = [0, 0, 1, 1]
    w = [1.0, 1.0, 1.0, 1.0]
    run_identity.emit(
        "handcalc_perfect",
        inputs={"p": p, "y": y, "w": w, "edges": [0.0, 0.5, 1.0]},
        expected_brier=0.01,
        decision_rule="Brier mean=0.01, REL=0, RES=0.25, UNC=0.25",
    )

    result = fixed_bins(p, y, w, edges=[0.0, 0.5, 1.0])

    assert result.n_nonempty_bins == 2
    assert result.n_empty_bins == 0
    assert_close(result.total_weight, 4.0)
    assert_close(result.effective_sample_size, 4.0)
    assert_close(result.brier, 0.01)
    assert_close(result.reliability, 0.0)
    assert_close(result.resolution, 0.25)
    assert_close(result.uncertainty, 0.25)
    assert_close(result.mean_forecast_variance_within_bin, 0.0025)
    assert_close(result.decomposition_residual, 0.0)
    assert result.weight_conservation_check.passed is True
    assert result.decomposition_check.passed is True


def test_overconfident_fixed_bins(run_identity):
    p = [0.9, 0.9, 0.1, 0.1]
    y = [0, 1, 0, 1]
    w = [1.0, 1.0, 1.0, 1.0]
    run_identity.emit(
        "handcalc_overconfident",
        inputs={"p": p, "y": y, "edges": [0.0, 0.5, 1.0]},
        expected_brier=0.41,
        expected_rel=0.16,
    )

    result = fixed_bins(p, y, w, edges=[0.0, 0.5, 1.0])

    assert_close(result.brier, 0.41)
    assert_close(result.reliability, 0.16)
    assert_close(result.resolution, 0.0)
    assert_close(result.uncertainty, 0.25)
    assert_close(result.mean_forecast_variance_within_bin, 0.0)
    assert_close(result.decomposition_residual, 0.0)


def test_constant_probability_reports_empty_equal_frequency_bins(run_identity):
    p = [0.5, 0.5, 0.5, 0.5]
    y = [0, 0, 1, 1]
    w = [1.0, 1.0, 1.0, 1.0]
    run_identity.emit(
        "handcalc_constant",
        n_bins=3,
        tie_policy="all tied probabilities remain in the first nonempty bin",
    )

    result = equal_frequency_bins(p, y, w, n_bins=3)

    assert result.n_nonempty_bins == 1
    assert result.n_empty_bins == 2
    assert result.bins[0].is_empty is False
    assert all(bin.is_empty for bin in result.bins[1:])
    assert_close(result.total_weight, 4.0)
    assert_close(result.brier, 0.25)
    assert_close(result.reliability, 0.0)
    assert_close(result.resolution, 0.0)
    assert_close(result.uncertainty, 0.25)
    assert_close(result.mean_forecast_variance_within_bin, 0.0)


def test_extreme_weights_use_kish_ess_and_preserve_total_weight(run_identity):
    p = [0.0, 0.0]
    y = [0, 1]
    w = [1.0, 999.0]
    run_identity.emit(
        "handcalc_extreme_weights",
        inputs={"p": p, "y": y, "w": w},
        expected_total_weight=1000.0,
        expected_ess=(1000.0**2) / (1.0 + 999.0**2),
        expected_y_mean=0.999,
    )

    fixed_result = fixed_bins(p, y, w, edges=[0.0, 0.5, 1.0])
    quantile_result = equal_frequency_bins(p, y, w, n_bins=2)

    for result in (fixed_result, quantile_result):
        assert result.n_nonempty_bins == 1
        assert result.n_empty_bins == 1
        assert_close(result.total_weight, 1000.0)
        assert_close(result.effective_sample_size, 1000.0**2 / (1.0 + 999.0**2), 1e-12)
        assert_close(result.outcome_rate, 0.999)
        assert_close(result.brier, 0.999)
        assert_close(result.reliability, 0.998001)
        assert_close(result.resolution, 0.0)
        assert_close(result.uncertainty, 0.000999)
        assert_close(result.decomposition_residual, 0.0)


@pytest.mark.parametrize(
    ("p", "y", "w", "n_bins"),
    [
        ([0.0, 0.1, 0.9, 1.0], [0, 0, 1, 1], [1, 1, 1, 1], 2),
        ([0.9, 0.9, 0.1, 0.1], [0, 1, 0, 1], [1, 1, 1, 1], 3),
        ([0.2, 0.2, 0.8], [0, 1, 1], [3.0, 2.0, 5.0], 2),
        ([0.5] * 8, [0, 0, 0, 0, 1, 1, 1, 1], [1.0] * 8, 3),
    ],
)
def test_equal_frequency_matches_independent_reference(p, y, w, n_bins, run_identity):
    run_identity.emit("reference_quantile_compare", n=len(p), n_bins=n_bins)
    actual = equal_frequency_bins(p, y, w, n_bins=n_bins, tolerance=1e-12)
    expected = quantile_diagnostics(p, y, w, n_bins)

    assert_close(actual.total_weight, expected["total_weight"])
    assert_close(actual.brier, expected["brier"])
    assert_close(actual.reliability, expected["reliability"])
    assert_close(actual.resolution, expected["resolution"])
    assert_close(actual.uncertainty, expected["uncertainty"])
    assert_close(actual.mean_forecast_variance_within_bin, expected["within_variance"])
    assert actual.n_nonempty_bins == sum(not bin_["empty"] for bin_ in expected["bins"])


def test_fixed_bins_match_independent_reference(run_identity):
    p = [0.0, 0.25, 0.5, 0.75, 1.0]
    y = [0, 1, 0, 1, 1]
    w = [1.0, 2.0, 0.5, 4.0, 3.0]
    run_identity.emit("reference_fixed_compare", n=len(p))
    actual = fixed_bins(p, y, w, edges=[0.0, 0.5, 1.0])
    expected = fixed_diagnostics(p, y, w, [0.0, 0.5, 1.0])

    assert_close(actual.total_weight, expected["total_weight"])
    assert_close(actual.brier, expected["brier"])
    assert_close(actual.effective_sample_size, expected["ess"])
    assert_close(actual.decomposition_residual, expected["residual"])
