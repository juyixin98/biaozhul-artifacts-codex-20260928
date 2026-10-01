"""Property-style cross checks against the independent stdlib reference."""

from __future__ import annotations

import random

import pytest

from app.kernel import equal_frequency_bins, fixed_bins
from tests.reference_stats import fixed_diagnostics, quantile_diagnostics


@pytest.mark.parametrize("seed", [17, 101, 20261001])
def test_random_weighted_fixed_and_quantile_statistics(seed, run_identity):
    generator = random.Random(seed)
    n = generator.randrange(12, 40)
    p = [generator.choice([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]) for _ in range(n)]
    y = [generator.randrange(2) for _ in range(n)]
    w = [generator.uniform(0.25, 5.0) for _ in range(n)]
    n_bins = generator.choice([2, 3, 5])
    run_identity.emit(
        "random_crosscheck",
        seed=seed,
        n=n,
        n_bins=n_bins,
        judgment="compare every Brier component to stdlib reference",
    )

    actual_fixed = fixed_bins(p, y, w, edges=[0.0, 0.25, 0.5, 0.75, 1.0])
    expected_fixed = fixed_diagnostics(p, y, w, [0.0, 0.25, 0.5, 0.75, 1.0])
    actual_quantile = equal_frequency_bins(p, y, w, n_bins=n_bins)
    expected_quantile = quantile_diagnostics(p, y, w, n_bins)

    for actual, expected in (
        (actual_fixed, expected_fixed),
        (actual_quantile, expected_quantile),
    ):
        assert actual.total_weight == pytest.approx(expected["total_weight"], abs=1e-11)
        assert actual.sample_count == expected["total_count"]
        assert actual.brier == pytest.approx(expected["brier"], abs=1e-12)
        assert actual.reliability == pytest.approx(expected["reliability"], abs=1e-12)
        assert actual.resolution == pytest.approx(expected["resolution"], abs=1e-12)
        assert actual.uncertainty == pytest.approx(expected["uncertainty"], abs=1e-12)
        assert actual.mean_forecast_variance_within_bin == pytest.approx(
            expected["within_variance"], abs=1e-12
        )
        assert actual.decomposition_residual == pytest.approx(0.0, abs=1e-12)
        assert actual.weight_conservation_check.passed is True
        assert actual.count_conservation_check.passed is True


def test_tied_probabilities_are_never_split_across_equal_frequency_bins(run_identity):
    p = [0.4, 0.4, 0.4, 0.6, 0.6]
    y = [0, 1, 0, 1, 0]
    w = [1.0, 1.0, 1.0, 1.0, 1.0]
    run_identity.emit("tie_non_splitting_check", p=p, n_bins=3)

    result = equal_frequency_bins(p, y, w, n_bins=3)
    nonempty = [bin_ for bin_ in result.bins if not bin_.is_empty]

    assert len(nonempty) == 2
    assert {bin_.count for bin in nonempty} == {3, 2}
    assert result.n_empty_bins == 1
    low_bin = next(bin_ for bin_ in result.bins if bin_.count == 3)
    assert low_bin.p_min == low_bin.p_max == 0.4
