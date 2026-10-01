"""Statistical kernel: exact and Monte Carlo permutation tests.

The kernel compares the means of two independent samples by permuting group
labels over sample *identities* (positions), never over distinct values, so
duplicated observations still occupy distinct slots in the permutation space.

Conventions fixed here (and relied upon by the API and the test-suite):

* test statistic: ``mean(group_a) - mean(group_b)``
* two-sided p: share of permutations with ``|T| >= |T_obs|``, using a
  relative tolerance of 1e-12 so float noise cannot flip borderline calls
* exact path: enumerates all ``C(n, k)`` label assignments, ``p = count / total``
* Monte Carlo path: plus-one corrected ``p = (1 + count) / (1 + B)`` with the
  binomial standard error of the raw proportion reported alongside
* cancellation: a stopped run reports ``status="cancelled"`` and
  ``p_value=None`` -- a partial sweep is never dressed up as a completed,
  non-significant result
"""
from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from itertools import combinations, islice
from typing import Callable, Literal, Sequence

import numpy as np
import scipy

from .errors import DomainError, ErrorCode

KERNEL_VERSION = "1.0.0"

Alternative = Literal["two-sided", "greater", "less"]
ALTERNATIVES: tuple[str, ...] = ("two-sided", "greater", "less")

STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
METHOD_EXACT = "exact"
METHOD_MONTE_CARLO = "monte_carlo"

_REL_TOL = 1e-12


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


@dataclass
class KernelResult:
    status: str
    method: str | None
    statistic: float
    p_value: float | None
    n_evaluated: int
    n_combinations: int
    mc_std_error: float | None
    alternative: str
    seed: int | None
    distribution: list[float] | None = None


def _extreme_mask(values: np.ndarray, observed: float, alternative: str) -> np.ndarray:
    tol = _REL_TOL * max(1.0, abs(observed))
    if alternative == "two-sided":
        return np.abs(values) >= abs(observed) - tol
    if alternative == "greater":
        return values >= observed - tol
    return values <= observed + tol


def permutation_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
    n_permutations: int | None = None,
    seed: int | None = None,
    exact_threshold: int = 50_000,
    max_evaluations: int | None = None,
    include_distribution: bool = False,
    should_stop: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    chunk_size: int = 2_048,
) -> KernelResult:
    """Run a two-sample permutation test on the difference of means.

    ``max_evaluations`` caps the number of permutations evaluated; hitting the
    cap (or a truthy ``should_stop``) yields a cancelled result. ``progress_cb``
    is invoked after every chunk with ``(evaluated, planned_total)``.
    """
    a = np.asarray(tuple(group_a), dtype=np.float64)
    b = np.asarray(tuple(group_b), dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise DomainError(ErrorCode.EMPTY_GROUP, "both groups need at least one observation")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise DomainError(ErrorCode.NON_FINITE_VALUE, "observations must be finite numbers")
    if alternative not in ALTERNATIVES:
        raise DomainError(ErrorCode.INVALID_PARAMETER, f"unknown alternative: {alternative!r}")
    if exact_threshold < 1:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "exact_threshold must be >= 1")

    combined = np.concatenate([a, b])
    n = combined.size
    k = a.size
    observed = float(a.sum()) / k - float(b.sum()) / (n - k)
    n_combinations = math.comb(n, k)

    if n_combinations <= exact_threshold:
        return _exact(
            combined, k, observed, alternative, n_combinations,
            max_evaluations, include_distribution, should_stop, progress_cb, chunk_size,
        )
    if n_permutations is None or n_permutations < 1:
        raise DomainError(
            ErrorCode.INVALID_PARAMETER,
            "n_permutations must be >= 1 for Monte Carlo runs",
        )
    return _monte_carlo(
        combined, k, observed, alternative, n_combinations, n_permutations,
        seed, max_evaluations, should_stop, progress_cb, chunk_size,
    )


def _exact(
    combined: np.ndarray,
    k: int,
    observed: float,
    alternative: str,
    n_combinations: int,
    max_evaluations: int | None,
    include_distribution: bool,
    should_stop: Callable[[], bool] | None,
    progress_cb: Callable[[int, int], None] | None,
    chunk_size: int,
) -> KernelResult:
    n = combined.size
    total = float(combined.sum())
    budget = n_combinations if max_evaluations is None else min(n_combinations, max_evaluations)
    extreme = 0
    evaluated = 0
    cancelled = False
    distribution: list[float] | None = [] if include_distribution else None
    combos = combinations(range(n), k)
    while evaluated < budget:
        if should_stop is not None and should_stop():
            cancelled = True
            break
        take = min(chunk_size, budget - evaluated)
        chunk = list(islice(combos, take))
        if not chunk:
            break
        idx = np.asarray(chunk, dtype=np.intp)
        sums_a = combined[idx].sum(axis=1)
        stats = sums_a / k - (total - sums_a) / (n - k)
        extreme += int(np.count_nonzero(_extreme_mask(stats, observed, alternative)))
        evaluated += idx.shape[0]
        if distribution is not None:
            distribution.extend(float(v) for v in stats)
        if progress_cb is not None:
            progress_cb(evaluated, n_combinations)
    cancelled = cancelled or evaluated < n_combinations
    if cancelled:
        return KernelResult(
            status=STATUS_CANCELLED, method=METHOD_EXACT, statistic=observed,
            p_value=None, n_evaluated=evaluated, n_combinations=n_combinations,
            mc_std_error=None, alternative=alternative, seed=None, distribution=None,
        )
    if distribution is not None:
        distribution.sort()
    return KernelResult(
        status=STATUS_COMPLETED, method=METHOD_EXACT, statistic=observed,
        p_value=extreme / n_combinations, n_evaluated=evaluated,
        n_combinations=n_combinations, mc_std_error=None, alternative=alternative,
        seed=None, distribution=distribution,
    )


def _monte_carlo(
    combined: np.ndarray,
    k: int,
    observed: float,
    alternative: str,
    n_combinations: int,
    n_permutations: int,
    seed: int | None,
    max_evaluations: int | None,
    should_stop: Callable[[], bool] | None,
    progress_cb: Callable[[int, int], None] | None,
    chunk_size: int,
) -> KernelResult:
    n = combined.size
    total = float(combined.sum())
    rng = np.random.default_rng(seed)
    budget = n_permutations if max_evaluations is None else min(n_permutations, max_evaluations)
    extreme = 0
    evaluated = 0
    cancelled = False
    while evaluated < budget:
        if should_stop is not None and should_stop():
            cancelled = True
            break
        take = min(chunk_size, budget - evaluated)
        draws = np.empty((take, k), dtype=np.intp)
        for i in range(take):
            draws[i] = rng.choice(n, size=k, replace=False)
        sums_a = combined[draws].sum(axis=1)
        stats = sums_a / k - (total - sums_a) / (n - k)
        extreme += int(np.count_nonzero(_extreme_mask(stats, observed, alternative)))
        evaluated += take
        if progress_cb is not None:
            progress_cb(evaluated, n_permutations)
    cancelled = cancelled or evaluated < n_permutations
    if cancelled or evaluated == 0:
        return KernelResult(
            status=STATUS_CANCELLED, method=METHOD_MONTE_CARLO, statistic=observed,
            p_value=None, n_evaluated=evaluated, n_combinations=n_combinations,
            mc_std_error=None, alternative=alternative, seed=seed, distribution=None,
        )
    p_raw = extreme / evaluated
    p_value = (extreme + 1) / (evaluated + 1)  # plus-one correction
    std_error = math.sqrt(p_raw * (1.0 - p_raw) / evaluated)
    return KernelResult(
        status=STATUS_COMPLETED, method=METHOD_MONTE_CARLO, statistic=observed,
        p_value=p_value, n_evaluated=evaluated, n_combinations=n_combinations,
        mc_std_error=std_error, alternative=alternative, seed=seed, distribution=None,
    )
