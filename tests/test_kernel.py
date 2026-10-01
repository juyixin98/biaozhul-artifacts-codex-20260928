from __future__ import annotations

import math

import numpy as np
import pytest

from app.errors import ComputationError, InvalidInputError
from app.stats.kernel import (
    beta_binomial_distribution,
    beta_binomial_log_pmf,
    credible_beta_interval,
    posterior_distribution,
    wilson_confidence_interval,
)
from experiments.oracle import (
    HAND_EXPECTATIONS,
    beta_credible_interval,
    normalized_beta_binomial,
)


def test_hand_computed_small_posterior_and_predictive_mass(assert_near, test_logger):
    expected = HAND_EXPECTATIONS["uniform_posterior_after_2_success_1_failure"]
    posterior = posterior_distribution(1.0, 1.0, 2, 1)
    assert_near(posterior.alpha, expected["posterior"][0])
    assert_near(posterior.beta, expected["posterior"][1])
    assert_near(posterior.mean, expected["mean"])
    assert_near(posterior.variance, expected["variance"])

    probabilities, log_probabilities, probability_sum = beta_binomial_distribution(
        2, posterior.alpha, posterior.beta
    )
    test_logger.note(
        "hand_small_counts",
        "posterior_and_predictive",
        {
            "posterior": [posterior.alpha, posterior.beta],
            "probabilities": [float(x) for x in probabilities],
            "sum": probability_sum,
        },
        "Uniform Beta(1,1) + (2,1) must give Beta(3,2); n=2 masses are 0.2,0.4,0.4.",
    )
    assert probabilities.tolist() == pytest.approx([0.2, 0.4, 0.4])
    assert math.isclose(probability_sum, 1.0, abs_tol=1e-12)
    assert np.all(np.isfinite(log_probabilities))


@pytest.mark.parametrize(
    ("alpha", "beta", "success", "failure", "trials"),
    [
        (1.0, 1.0, 10_000, 0, 20),
        (0.5, 0.5, 10**10, 10**10, 200),
        (1.0e100, 1.0, 1_000_000, 0, 10),
        (1.0e-100, 1.0e-100, 0, 0, 5),
    ],
)
def test_extreme_parameters_are_finite_and_normalize(
    alpha, beta, success, failure, trials, test_logger
):
    posterior = posterior_distribution(alpha, beta, success, failure)
    probabilities, log_probabilities, probability_sum = beta_binomial_distribution(
        trials, posterior.alpha, posterior.beta
    )
    oracle_probabilities, oracle_sum = normalized_beta_binomial(
        trials, posterior.alpha, posterior.beta
    )
    test_logger.note(
        "extreme_beta_binomial",
        "compared_with_independent_oracle",
        {
            "posterior_alpha": posterior.alpha,
            "posterior_beta": posterior.beta,
            "trials": trials,
            "sum": probability_sum,
            "oracle_sum": oracle_sum,
            "max_abs_difference": float(np.max(np.abs(probabilities - oracle_probabilities))),
        },
        "All log masses are finite and normalized probabilities agree with the independent beta-function form.",
    )
    assert np.all(np.isfinite(log_probabilities))
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert math.isclose(probability_sum, 1.0, abs_tol=1e-10)
    assert probabilities.tolist() == pytest.approx(oracle_probabilities.tolist(), abs=1e-11)


def test_zero_trials_predictive_distribution():
    probabilities, _logs, probability_sum = beta_binomial_distribution(0, 3.0, 2.0)
    assert probabilities.tolist() == [1.0]
    assert probability_sum == 1.0


def test_credible_interval_terms_are_explicit_and_extremes_are_bounded():
    lower, upper = credible_beta_interval(3.0, 2.0)
    oracle_lower, oracle_upper = beta_credible_interval(3.0, 2.0)
    assert lower == pytest.approx(oracle_lower, abs=1e-12)
    assert upper == pytest.approx(oracle_upper, abs=1e-12)
    assert lower > 0.0 and upper < 1.0

    extreme_lower, extreme_upper = credible_beta_interval(1.0e100 + 1.0, 2.0)
    assert 0.0 <= extreme_lower <= extreme_upper <= 1.0
    assert math.isfinite(extreme_lower) and math.isfinite(extreme_upper)

    near_one_lower, near_one_upper = credible_beta_interval(1.0e6 + 1.0, 1.0)
    assert near_one_lower == pytest.approx(0.9999963111242348, abs=1e-15)
    assert near_one_upper == pytest.approx(0.9999999746822174, abs=1e-15)


def test_wilson_frequentist_interval_is_distinct_from_credible_interval():
    lower, upper = wilson_confidence_interval(2, 2, 0.95)
    assert lower == pytest.approx(0.34238022750665303, abs=1e-12)
    assert upper == pytest.approx(1.0, abs=1e-12)
    assert wilson_confidence_interval(0, 0) == (None, None)


@pytest.mark.parametrize(
    ("alpha", "beta", "category"),
    [
        (0.0, 1.0, "strictly positive"),
        (-1.0, 1.0, "strictly positive"),
        (float("inf"), 1.0, "finite"),
        (float("nan"), 1.0, "finite"),
        (True, 1.0, "numeric"),
    ],
)
def test_invalid_prior_failure_category(alpha, beta, category):
    with pytest.raises(InvalidInputError) as exc_info:
        beta_binomial_log_pmf(0, 1, alpha, beta)
    assert exc_info.value.category == "invalid_input"
    assert category in str(exc_info.value)


@pytest.mark.parametrize(
    ("success", "failure"),
    [(-1, 0), (0, -1), (1.5, 0), (True, 0)],
)
def test_invalid_counts_are_invalid_input(success, failure):
    with pytest.raises(InvalidInputError) as exc_info:
        posterior_distribution(1.0, 1.0, success, failure)
    assert exc_info.value.category == "invalid_input"


def test_unsupported_prior_bound_is_a_distinct_input_failure():
    with pytest.raises(InvalidInputError) as exc_info:
        posterior_distribution(1.0e101, 1.0, 0, 0)
    assert exc_info.value.code == "INVALID_INPUT"
    assert "supported bound" in str(exc_info.value)


def test_computation_failure_is_detectable_when_log_pmf_non_finite(monkeypatch):
    def broken_gammaln(_values):
        return np.asarray([np.inf])

    monkeypatch.setattr("app.stats.kernel.gammaln", broken_gammaln)
    with np.errstate(all="ignore"):
        with pytest.raises(ComputationError) as exc_info:
            beta_binomial_distribution(0, 1.0, 1.0)
    assert exc_info.value.category == "computation_failed"
