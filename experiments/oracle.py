"""Independent Beta-Binomial reference formulas.

This module intentionally does not import app.stats. Tests use it as an
independent oracle, while the smallest cases also hard-code hand-computed
numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class BetaPosteriorOracle:
    alpha: float
    beta: float
    mean: float
    variance: float


def beta_posterior(
    prior_alpha: float,
    prior_beta: float,
    successes: int,
    failures: int,
) -> BetaPosteriorOracle:
    alpha = prior_alpha + successes
    beta_param = prior_beta + failures
    total = alpha + beta_param
    return BetaPosteriorOracle(
        alpha=alpha,
        beta=beta_param,
        mean=alpha / total,
        variance=alpha * beta_param / (total * total * (total + 1.0)),
    )


def beta_binomial_log_pmf_alt(
    k: int | np.ndarray, n: int, alpha: float, beta_param: float
) -> np.ndarray:
    """Relative recurrence for Beta-Binomial weights, then a log-normalizer."""
    k_array = np.asarray(k, dtype=np.int64)
    relative_log_weights = np.zeros(n + 1, dtype=np.float64)
    for k in range(n):
        denominator_value = beta_param + n - k - 1
        numerator = alpha + k
        if numerator <= 0.0 or denominator_value <= 0.0:
            relative_log_weights[k + 1 : -1] = -np.inf
        else:
            log_ratio = (
                np.log(n - k)
                - np.log(k + 1)
                + np.log(numerator)
                - np.log(denominator_value)
            )
            relative_log_weights[k + 1] = relative_log_weights[k] + log_ratio
    normalized = relative_log_weights - logsumexp(relative_log_weights)
    result = normalized[k_array]
    return np.asarray(result, dtype=np.float64)


def normalized_beta_binomial(
    n: int, alpha: float, beta_param: float
) -> tuple[np.ndarray, float]:
    k = np.arange(n + 1, dtype=np.int64)
    log_probabilities = beta_binomial_log_pmf_alt(k, n, alpha, beta_param)
    probabilities = np.exp(log_probabilities)
    return probabilities, float(math.fsum(float(x) for x in probabilities))


def beta_credible_interval(
    alpha: float, beta_param: float, level: float = 0.95
) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    values = beta_dist.ppf([tail, 1.0 - tail], alpha, beta_param)
    return float(values[0]), float(values[1])


def wilson_95(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p_hat = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p_hat + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


HAND_EXPECTATIONS = {
    "uniform_posterior_after_2_success_1_failure": {
        "prior": (1.0, 1.0),
        "counts": (2, 1),
        "posterior": (3.0, 2.0),
        "mean": 0.6,
        "variance": 0.04,
    },
    "predictive_n_1": {
        "probabilities": [0.4, 0.6],
        "sum": 1.0,
    },
    "predictive_n_2": {
        # C(2,k) B(k+3,2-k+2)/B(3,2): 6/30, 12/30, 12/30
        "probabilities": [0.2, 0.4, 0.4],
        "sum": 1.0,
    },
}
