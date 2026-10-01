from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from app.errors import ComputationError, InvalidInputError


@dataclass(frozen=True)
class Posterior:
    alpha: float
    beta: float
    mean: float
    variance: float


def validate_prior(alpha: float, beta: float, maximum: float = 1.0e100) -> None:
    for name, value in (("alpha", alpha), ("beta", beta)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise InvalidInputError(f"Prior {name} must be numeric.", {"field": name})
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise InvalidInputError(
                f"Prior {name} must be finite and strictly positive.", {"field": name}
            )
        if numeric > maximum:
            raise InvalidInputError(
                f"Prior {name} exceeds the supported bound {maximum!r}.",
                {"field": name, "value": numeric, "maximum": maximum},
            )


def validate_counts(
    success_count: int,
    failure_count: int,
    maximum: int | None = None,
    *,
    names: tuple[str, str] = ("success_count", "failure_count"),
) -> None:
    for name, value in zip(names, (success_count, failure_count), strict=True):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidInputError(f"{name} must be a non-negative integer.", {"field": name})
        if value < 0:
            raise InvalidInputError(f"{name} must be non-negative.", {"field": name})
        if maximum is not None and value > maximum:
            raise InvalidInputError(
                f"{name} exceeds the supported bound {maximum}.",
                {"field": name, "value": value, "maximum": maximum},
            )


def posterior_parameters(
    prior_alpha: float,
    prior_beta: float,
    success_count: int,
    failure_count: int,
) -> tuple[float, float]:
    validate_prior(prior_alpha, prior_beta)
    validate_counts(success_count, failure_count)
    return float(prior_alpha) + float(success_count), float(prior_beta) + float(failure_count)


def posterior_distribution(
    prior_alpha: float,
    prior_beta: float,
    success_count: int,
    failure_count: int,
) -> Posterior:
    alpha, beta_param = posterior_parameters(prior_alpha, prior_beta, success_count, failure_count)
    total = alpha + beta_param
    mean = alpha / total
    # Factor the denominator to avoid avoidable overflow for large posterior values.
    variance = (alpha / total) * (beta_param / total) / (total + 1.0)
    return Posterior(alpha=alpha, beta=beta_param, mean=mean, variance=variance)


def beta_binomial_log_pmf(k: np.ndarray | int, n: int, alpha: float, beta_param: float) -> np.ndarray:
    """Return the Beta-Binomial log PMF with stable integer rising products."""
    validate_prior(alpha, beta_param)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise InvalidInputError("trials must be a non-negative integer.", {"field": "trials"})
    k_array = np.asarray(k, dtype=np.int64)
    if np.any(k_array < 0) or np.any(k_array > n):
        raise InvalidInputError("Each k must satisfy 0 <= k <= trials.", {"trials": n})

    def rising_logs(value: float) -> np.ndarray:
        # Γ(value + m) / Γ(value) = product_{j=0}^{m-1}(value + j) for integer m.
        # Summing individual logs avoids catastrophic cancellation in
        # gammaln(value+m)-gammaln(value) when value is around 1e10 or larger.
        cumulative = np.zeros(n + 1, dtype=np.float64)
        if n > 0:
            terms = np.log(value + np.arange(n, dtype=np.float64))
            if not np.all(np.isfinite(terms)):
                raise ComputationError("A rising-product term was not finite.")
            cumulative[1:] = np.cumsum(terms)
        return cumulative

    rising_alpha = rising_logs(alpha)
    rising_beta = rising_logs(beta_param)
    rising_total = rising_logs(alpha + beta_param)

    log_choose = gammaln(n + 1) - gammaln(k_array + 1) - gammaln(n - k_array + 1)
    log_pmf = (
        log_choose
        + rising_alpha[k_array]
        + rising_beta[n - k_array]
        - rising_total[n]
    )
    return np.asarray(log_pmf, dtype=np.float64)


def beta_binomial_distribution(
    n: int, alpha: float, beta_param: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return probabilities, normalized log probabilities, and their finite sum."""
    k = np.arange(n + 1, dtype=np.int64)
    raw_log_pmf = beta_binomial_log_pmf(k, n, alpha, beta_param)
    if not np.all(np.isfinite(raw_log_pmf)):
        raise ComputationError("Beta-Binomial log probabilities contained non-finite values.")
    log_normalizer = float(logsumexp(raw_log_pmf))
    if not math.isfinite(log_normalizer):
        raise ComputationError("Beta-Binomial normalizer was not finite.")
    log_pmf = raw_log_pmf - log_normalizer
    probabilities = np.exp(log_pmf)
    probability_sum = float(math.fsum(float(x) for x in probabilities))
    if probability_sum <= 0.0 or not math.isfinite(probability_sum):
        raise ComputationError("Beta-Binomial probability mass did not normalize.")
    return probabilities, log_pmf, probability_sum


def _normal_interval(alpha: float, beta_param: float, tail: float) -> tuple[float, float]:
    total = alpha + beta_param
    mean = alpha / total
    variance = (alpha / total) * (beta_param / total) / (total + 1.0)
    z = norm.ppf([tail, 1.0 - tail])
    endpoints = np.asarray(mean + z * math.sqrt(max(variance, 0.0)), dtype=np.float64)
    if not np.all(np.isfinite(endpoints)):
        raise ComputationError("Credible interval normal fallback was not finite.")
    lower, upper = (float(x) for x in endpoints)
    return min(1.0, max(0.0, lower)), min(1.0, max(0.0, upper))


def credible_beta_interval(
    alpha: float,
    beta_param: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Posterior Beta equal-tailed credible interval.

    SciPy's beta quantile routine can collapse for extreme power-law or very
    concentrated posteriors. Exact power-law tail forms and a log-safe normal
    fallback keep those inputs finite without raising overflow.
    """
    validate_prior(alpha, beta_param)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise InvalidInputError("interval level must be between 0 and 1.", {"level": level})
    tail = (1.0 - level) / 2.0

    # Below this scale, finite floating-point endpoints cannot distinguish the
    # enormous probability mass very close to a boundary. Report the full range.
    if min(alpha, beta_param) <= 1.0e-4:
        return 0.0, 1.0

    # In very asymmetric posteriors, evaluate the small complementary tail in
    # Gamma scale and transform stably. For Beta(a,b), a >> b:
    # a(1-X) is asymptotically Gamma(b,1); the b >> a case is symmetric.
    if beta_param > 0.0 and alpha / beta_param >= 1.0e5:
        gamma_tails = gamma_dist.ppf([tail, 1.0 - tail], beta_param)
        # X = 1 - D, with D approximately Gamma(shape=b, scale=1/a).
        lower = 1.0 - float(gamma_tails[1]) / alpha
        upper = 1.0 - float(gamma_tails[0]) / alpha
        return min(1.0, max(0.0, lower)), min(1.0, max(0.0, upper))
    if alpha > 0.0 and beta_param / alpha >= 1.0e5:
        gamma_tails = gamma_dist.ppf([tail, 1.0 - tail], alpha)
        # X is approximately Gamma(shape=a, scale=1/b) near zero.
        lower = float(gamma_tails[0]) / beta_param
        upper = float(gamma_tails[1]) / beta_param
        return (
            min(1.0, max(0.0, lower)),
            min(1.0, max(0.0, upper)),
        )

    quantiles = beta_dist.ppf([tail, 1.0 - tail], alpha, beta_param)
    if np.all(np.isfinite(quantiles)):
        lower, upper = float(quantiles[0]), float(quantiles[1])
        if 0.0 <= lower <= upper <= 1.0:
            return lower, upper

    return _normal_interval(alpha, beta_param, tail)


def wilson_confidence_interval(
    successes: int,
    trials: int,
    level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Frequentist Wilson score interval; deliberately distinct from a posterior."""
    validate_counts(successes, trials, names=("successes", "trials"))
    if successes > trials:
        raise InvalidInputError("successes cannot exceed trials.", {"successes": successes, "trials": trials})
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise InvalidInputError("interval level must be between 0 and 1.", {"level": level})
    if trials == 0:
        return None, None

    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    p_hat = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p_hat + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
