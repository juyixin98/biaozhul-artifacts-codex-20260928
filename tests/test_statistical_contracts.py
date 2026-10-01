from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from app.errors import ComputationError, InvalidInputError
from app.stats import (
    beta_binomial_predictions,
    posterior_parameters,
    wilson_confidence_interval,
)
from tests.replay_logger import ReplayLogger


def test_hand_computed_small_posterior_and_predictions(
    replay_logger: ReplayLogger,
) -> None:
    post_alpha, post_beta = posterior_parameters(1, 1, 4, 5)
    replay_logger.state(
        "posterior",
        {"alpha": post_alpha, "beta": post_beta},
        "Beta(1,1) plus four successes and five failures is Beta(5,6).",
    )
    assert (post_alpha, post_beta) == (5, 6)

    distribution = beta_binomial_predictions(3, post_alpha, post_beta)
    observed = [point.probability for point in distribution.points]
    # Hand-derived for Beta(5,6), n=3:
    # P(k)=C(3,k) B(5+k,6+3-k)/B(5,6).
    expected = [28 / 143, 105 / 286, 45 / 143, 35 / 286]
    replay_logger.state(
        "predictive_distribution",
        {"observed": observed, "expected": expected, "sum": sum(observed)},
        "Each mass is C(3,k) B(5+k,6+3-k)/B(5,6).",
    )
    assert observed == pytest.approx(expected, abs=1e-15)
    assert math.fsum(observed) == pytest.approx(1.0, abs=1e-15)


def test_all_success_boundary_and_zero_count(replay_logger: ReplayLogger) -> None:
    post_alpha, post_beta = posterior_parameters(1, 1, 10, 0)
    prediction = beta_binomial_predictions(1, post_alpha, post_beta)
    probabilities = [point.probability for point in prediction.points]
    replay_logger.state(
        "all_successes",
        {"posterior": [post_alpha, post_beta], "probabilities": probabilities},
        "With zero observed failures, future failure probability is 1/12.",
    )
    assert (post_alpha, post_beta) == (11, 1)
    assert probabilities == pytest.approx([1 / 12, 11 / 12], abs=1e-15)

    zero = beta_binomial_predictions(0, 1, 1)
    assert zero.points[0].future_successes == 0
    assert zero.points[0].probability == pytest.approx(1.0)


def test_large_counts_match_independent_scipy_reference(
    replay_logger: ReplayLogger,
) -> None:
    successes, failures, future_n = 1_000_000, 2_000_000, 100
    post_alpha, post_beta = posterior_parameters(
        1, 1, successes, failures
    )
    distribution = beta_binomial_predictions(future_n, post_alpha, post_beta)
    observed = np.array([point.probability for point in distribution.points])
    expected = stats.betabinom.pmf(
        np.arange(future_n + 1), future_n, post_alpha, post_beta
    )
    replay_logger.state(
        "large_reference_comparison",
        {
            "posterior": [post_alpha, post_beta],
            "max_abs_error": float(np.max(np.abs(observed - expected))),
            "sum": float(observed.sum()),
        },
        "Expected distribution comes directly from scipy.stats.betabinom.",
    )
    nonzero = expected > 1e-300
    relative_error = np.max(np.abs(observed[nonzero] - expected[nonzero]) / expected[nonzero])
    assert relative_error < 1e-9
    assert observed.sum() == pytest.approx(1.0, abs=1e-15)


def test_extreme_prior_does_not_overflow_predictive_log_mass(
    replay_logger: ReplayLogger,
) -> None:
    extreme = 1.0e200
    distribution = beta_binomial_predictions(10, extreme, extreme)
    probabilities = np.array([point.probability for point in distribution.points])
    replay_logger.state(
        "extreme_prior",
        {"all_finite": bool(np.isfinite(probabilities).all()),
         "sum": float(probabilities.sum())},
        "Log-gamma differences use a Stirling difference instead of subtracting infinities.",
    )
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-15)
    assert probabilities[5] == pytest.approx(max(probabilities), abs=1e-15)


def test_sample_contract_rejects_non_integer_or_negative_counts() -> None:
    with pytest.raises(InvalidInputError, match="non-negative"):
        posterior_parameters(1, 1, -1, 1)
    with pytest.raises(InvalidInputError, match="integer"):
        posterior_parameters(1, 1, -1.0, 1)
    with pytest.raises(InvalidInputError, match="integer"):
        posterior_parameters(1, 1, True, 1)
    with pytest.raises(InvalidInputError, match="strictly positive"):
        posterior_parameters(0, 1, 0, 0)


def test_credible_and_confidence_intervals_are_separate_contracts() -> None:
    # The production credible interval uses a beta quantile; this expected value
    # is independently computed here from scipy.stats.beta.
    credible_quantiles = stats.beta.ppf([0.025, 0.975], 5, 6)
    wilson = wilson_confidence_interval(4, 5, 0.95)
    assert wilson is not None
    assert wilson.lower < 4 / 9 < wilson.upper
    assert credible_quantiles[0] < 4 / 9 < credible_quantiles[1]
    assert wilson_confidence_interval(0, 0) is None


def test_computation_failure_is_distinct_from_input_error(monkeypatch) -> None:
    def broken_pmf(*args, **kwargs):
        raise ComputationError("forced numerical failure", {"point": "kernel"})

    monkeypatch.setattr("app.stats.beta_binomial_log_pmf", broken_pmf)
    with pytest.raises(ComputationError, match="forced numerical failure"):
        beta_binomial_predictions(2, 1, 1)
