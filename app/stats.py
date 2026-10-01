"""Numerically stable Beta-Binomial statistical kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln, logsumexp

from app.domain import ensure_count, ensure_future_n, ensure_level, ensure_prior
from app.errors import ComputationError


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    level: float


@dataclass(frozen=True)
class PredictionPoint:
    future_successes: int
    probability: float


@dataclass(frozen=True)
class PredictionDistribution:
    future_n: int
    points: list[PredictionPoint]
    mean: float
    variance: float


def posterior_parameters(
    alpha: object, beta: object, successes: object, failures: object
) -> tuple[float, float]:
    """Return Beta posterior parameters after one sufficiency update."""
    post_alpha, post_beta = ensure_prior(alpha, beta)
    successes = ensure_count(successes, "successes")
    failures = ensure_count(failures, "failures")
    return post_alpha + successes, post_beta + failures


def beta_moments(alpha: float, beta: float) -> tuple[float, float]:
    """Return Beta mean and variance without multiplying two large shapes."""
    total = alpha + beta
    mean = alpha / total
    variance = mean * (1.0 - mean) / (total + 1.0)
    return mean, variance


def beta_credible_interval(
    alpha: float, beta: float, level: float = 0.95
) -> Interval:
    """Equal-tailed credible interval for latent p, not a confidence interval."""
    level = ensure_level(level)
    tail = (1.0 - level) / 2.0
    quantiles = stats.beta.ppf([tail, 1.0 - tail], alpha, beta)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if not np.all(np.isfinite(quantiles)) or quantiles[0] > quantiles[1]:
        raise ComputationError(
            "Beta credible interval could not be computed.",
            {"alpha": alpha, "beta": beta, "level": level},
        )
    return Interval(float(quantiles[0]), float(quantiles[1]), level)


def wilson_confidence_interval(
    successes: int, failures: int, level: float = 0.95
) -> Interval | None:
    """Frequentist Wilson interval for the observed binomial proportion.

    Return None when there are no observations; a zero-width interval at 0 or 1
    would hide the fact that a Wilson interval is undefined without data.
    """
    successes = ensure_count(successes, "successes")
    failures = ensure_count(failures, "failures")
    level = ensure_level(level)
    n = successes + failures
    if n == 0:
        return None

    z = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return Interval(center - half, center + half, level)


def _delta_log_gamma(x: float, d: int | np.ndarray) -> float | np.ndarray:
    """Compute log Gamma(x+d) - log Gamma(x) stably.

    scipy's direct log-gamma calls are accurate while they remain finite. The
    rising-factorial sum is reserved for huge x where direct calls overflow.
    """
    d_array = np.asarray(d, dtype=np.int64)
    use_stable = (x >= 1.0e15) & (d_array <= x)

    result = np.empty(np.shape(d_array), dtype=np.float64)
    direct = ~use_stable
    if np.any(direct):
        result[direct] = gammaln(x + d_array[direct]) - gammaln(x)

    if np.any(use_stable):
        dd = d_array[use_stable]
        values = np.empty(dd.shape, dtype=np.float64)
        for index, d_value in enumerate(dd):
            d_int = int(d_value)
            if d_int == 0:
                values[index] = 0.0
            else:
                # log x(x+1)...(x+d-1) plus the difference of Stirling's
                # 1/(12z) correction. This never evaluates log Gamma(1e200).
                values[index] = (
                    np.sum(np.log(x + np.arange(d_int, dtype=np.float64)))
                    + 1.0 / (12.0 * (x + d_int))
                    - 1.0 / (12.0 * x)
                )
        result[use_stable] = values

    if not np.all(np.isfinite(result)):
        raise ComputationError(
            "Non-finite log-gamma difference encountered.",
            {"x": x, "d_min": int(np.min(d_array)), "d_max": int(np.max(d_array))},
        )
    if np.ndim(d) == 0:
        return float(result[()])
    return result


def beta_binomial_log_pmf(
    future_n: object,
    k: int | np.ndarray,
    post_alpha: float,
    post_beta: float,
) -> float | np.ndarray:
    """Log Beta-Binomial predictive mass using gamma-ratio differences."""
    future_n = ensure_count(future_n, "future_n")
    k_array = np.asarray(k, dtype=np.int64)
    if np.any(k_array < 0) or np.any(k_array > future_n):
        raise ComputationError(
            "Prediction support point is outside [0, future_n].",
            {
                "future_n": future_n,
                "k_min": int(np.min(k_array)),
                "k_max": int(np.max(k_array)),
            },
        )

    failures = np.asarray(future_n - k_array, dtype=np.int64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        log_binomial = (
            gammaln(future_n + 1)
            - gammaln(k_array + 1)
            - gammaln(failures + 1)
        )
        can_use_betaln = np.isfinite(
            gammaln(post_alpha + post_beta + future_n)
        )
        if can_use_betaln:
            values = log_binomial + betaln(
                post_alpha + k_array, post_beta + failures
            ) - betaln(post_alpha, post_beta)
        else:
            values = (
                log_binomial
                + _delta_log_gamma(post_alpha, k_array)
                + _delta_log_gamma(post_beta, failures)
                - _delta_log_gamma(post_alpha + post_beta, future_n)
            )
    if not np.all(np.isfinite(values)):
        raise ComputationError(
            "Non-finite Beta-Binomial log probability encountered.",
            {"future_n": future_n, "alpha": post_alpha, "beta": post_beta},
        )
    if np.ndim(k) == 0:
        return float(values[()])
    return values


def beta_binomial_predictions(
    future_n: object, post_alpha: float, post_beta: float
) -> PredictionDistribution:
    """Return the complete normalized predictive distribution.

    Log probabilities are shifted by their maximum before exponentiation, and
    the final vector is explicitly normalized. This controls tiny round-off
    error and also makes the sum-one contract easy to test.
    """
    future_n = ensure_future_n(future_n)
    if not (math.isfinite(post_alpha) and post_alpha > 0):
        raise ComputationError(
            "Posterior alpha must be finite and positive.", {"alpha": post_alpha}
        )
    if not (math.isfinite(post_beta) and post_beta > 0):
        raise ComputationError(
            "Posterior beta must be finite and positive.", {"beta": post_beta}
        )

    k = np.arange(future_n + 1, dtype=np.int64)
    log_pmf = np.asarray(
        beta_binomial_log_pmf(future_n, k, post_alpha, post_beta), dtype=np.float64
    )
    log_pmf = log_pmf - logsumexp(log_pmf)
    probabilities = np.exp(log_pmf, dtype=np.float64)
    probability_sum = float(np.sum(probabilities))
    if probability_sum <= 0.0 or not math.isfinite(probability_sum):
        raise ComputationError(
            "Predictive distribution could not be normalized.",
            {"sum": probability_sum},
        )
    probabilities /= probability_sum

    total = post_alpha + post_beta
    p = post_alpha / total
    mean = future_n * p
    variance = future_n * p * (1.0 - p) * (total + future_n) / (total + 1.0)
    points = [
        PredictionPoint(future_successes=int(kk), probability=float(pp))
        for kk, pp in zip(k, probabilities)
    ]
    return PredictionDistribution(
        future_n=future_n,
        points=points,
        mean=float(mean),
        variance=float(variance),
    )
