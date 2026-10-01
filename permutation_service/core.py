"""Statistical kernel for a two-sided difference of independent sample means.

The reference statistic is abs(mean(group 1) - mean(group 2)).  Exact tests
enumerate group-size-preserving assignments by sample *identity*.  Therefore,
equal measured values still create distinct assignments and contribute with
their multiplicity.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.special import comb

from .errors import (
    CancellationError,
    ComputationError,
    ExactLimitExceededError,
    InvalidInputError,
)

Method = Literal["auto", "exact", "monte_carlo"]
ProgressCallback = Callable[[int, int, str, int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class PermutationRequest:
    values: list[float]
    group1_size: int
    alpha: float = 0.05
    method: Method = "auto"
    monte_carlo_replications: int = 9_999
    exact_max_combinations: int = 100_000
    exact_limit_cap: int = 100_000_000
    max_values: int = 10_000
    max_monte_carlo_replications: int = 10_000_000
    random_seed: int | None = None
    return_null_distribution: bool = False


@dataclass(frozen=True)
class PermutationResult:
    statistic: float
    group1_mean: float
    group2_mean: float
    p_value: float
    significant: bool
    alpha: float
    method_used: Literal["exact", "monte_carlo"]
    alternative: Literal["two-sided"]
    total_possible_assignments: int
    evaluated_permutations: int
    extreme_permutations: int
    random_seed: int | None
    standard_error: float | None
    confidence_interval_95: tuple[float, float] | None
    null_distribution: tuple[float, ...] = field(default_factory=tuple)
    observed_assignment_excluded: bool = False
    finite_sample_correction: str | None = None


def _check_request(request: PermutationRequest) -> np.ndarray:
    try:
        values = np.asarray(request.values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError("values must be numeric") from exc

    if values.ndim != 1:
        raise InvalidInputError("values must be a one-dimensional list")
    n = values.size
    if n < 2:
        raise InvalidInputError("at least two observations are required")
    if n > request.max_values:
        raise InvalidInputError(
            f"number of values must not exceed {request.max_values}",
            details={"received": n, "limit": request.max_values},
        )
    if not np.all(np.isfinite(values)):
        raise InvalidInputError("all values must be finite")
    if not 1 <= request.group1_size < n:
        raise InvalidInputError(
            "group1_size must be between 1 and len(values)-1",
            details={"group1_size": request.group1_size, "n": n},
        )
    if not 0.0 < request.alpha < 1.0:
        raise InvalidInputError("alpha must be strictly between 0 and 1")
    if request.method not in {"auto", "exact", "monte_carlo"}:
        raise InvalidInputError("method must be auto, exact, or monte_carlo")
    if request.monte_carlo_replications < 1:
        raise InvalidInputError("monte_carlo_replications must be positive")
    if request.monte_carlo_replications > request.max_monte_carlo_replications:
        raise InvalidInputError(
            "monte_carlo_replications exceeds configured limit",
            details={
                "received": request.monte_carlo_replications,
                "limit": request.max_monte_carlo_replications,
            },
        )
    if request.exact_max_combinations < 0:
        raise InvalidInputError("exact_max_combinations cannot be negative")
    if request.exact_limit_cap < 0:
        raise InvalidInputError("exact_limit_cap cannot be negative")
    if request.random_seed is not None and not isinstance(request.random_seed, int):
        raise InvalidInputError("random_seed must be an integer")
    return values


def _statistic(values: np.ndarray, group1_indices) -> float:
    indices = np.asarray(group1_indices, dtype=np.intp)
    n = values.size
    k = indices.size
    group1_sum = float(values[indices].sum())
    group2_sum = float(values.sum() - group1_sum)
    return abs(group1_sum / k - group2_sum / (n - k))


def _raise_if_canceled(is_canceled: CancelCallback | None) -> None:
    if is_canceled is not None and is_canceled():
        raise CancellationError("permutation computation was canceled")


def _emit(
    progress: ProgressCallback | None,
    completed: int,
    total: int,
    phase: str,
    batch_size: int,
) -> None:
    if progress is not None:
        progress(completed, total, phase, batch_size)


def _exact(
    values: np.ndarray,
    group1_size: int,
    request: PermutationRequest,
    progress: ProgressCallback | None,
    is_canceled: CancelCallback | None,
    *,
    guarded: bool,
) -> PermutationResult:
    n = values.size
    total = int(comb(n, group1_size, exact=True))
    if guarded and total > request.exact_limit_cap:
        raise ExactLimitExceededError(
            "exact enumeration exceeds the configured safety limit",
            details={
                "total_possible_assignments": total,
                "limit": request.exact_limit_cap,
            },
        )

    observed = _statistic(values, range(group1_size))
    iterator = itertools.combinations(range(n), group1_size)
    batch_size = 4_096
    extreme = 0
    null_values: list[float] = []
    completed = 0

    _raise_if_canceled(is_canceled)
    _emit(progress, 0, total, "exact-enumeration", 0)
    while completed < total:
        _raise_if_canceled(is_canceled)
        chunk = list(itertools.islice(iterator, batch_size))
        for assignment in chunk:
            statistic = _statistic(values, assignment)
            extreme += int(statistic >= observed)
            if request.return_null_distribution:
                null_values.append(statistic)
        completed += len(chunk)
        _emit(progress, completed, total, "exact-enumeration", len(chunk))
        if not chunk:  # Defensive guard against iterator/count disagreement.
            raise ComputationError("exact combination iterator ended prematurely")

    p_value = extreme / total
    return PermutationResult(
        statistic=observed,
        group1_mean=float(values[:group1_size].mean()),
        group2_mean=float(values[group1_size:].mean()),
        p_value=p_value,
        significant=p_value <= request.alpha,
        alpha=request.alpha,
        method_used="exact",
        alternative="two-sided",
        total_possible_assignments=total,
        evaluated_permutations=total,
        extreme_permutations=extreme,
        random_seed=None,
        standard_error=None,
        confidence_interval_95=None,
        null_distribution=tuple(null_values),
        observed_assignment_excluded=False,
        finite_sample_correction=None,
    )


def _monte_carlo(
    values: np.ndarray,
    group1_size: int,
    request: PermutationRequest,
    progress: ProgressCallback | None,
    is_canceled: CancelCallback | None,
    seed: int,
) -> PermutationResult:
    n = values.size
    total = math.comb(n, group1_size)
    if total <= 1:
        raise InvalidInputError("Monte Carlo sampling requires at least two assignments")

    rng = np.random.default_rng(seed)
    observed = _statistic(values, range(group1_size))
    base_labels = np.zeros(n, dtype=bool)
    base_labels[:group1_size] = True
    observed_labels = base_labels.copy()

    requested = request.monte_carlo_replications
    extreme = 0
    sampled = 0
    null_values: list[float] = []
    vector_batch = 1_024

    _raise_if_canceled(is_canceled)
    _emit(progress, 0, requested, "monte-carlo-sampling", 0)
    while sampled < requested:
        _raise_if_canceled(is_canceled)
        remaining = requested - sampled
        batch_size = min(vector_batch, remaining)
        labels = np.tile(base_labels, (batch_size, 1))
        # A fresh permutation independently shuffles sample identities for
        # every replicate while leaving exactly group1_size labels true.
        labels = rng.permutation(labels, axis=1)
        is_observed = np.all(labels == observed_labels, axis=1)

        for row in labels[~is_observed]:
            statistic = _statistic(values, np.flatnonzero(row))
            extreme += int(statistic >= observed)
            sampled += 1
            if request.return_null_distribution:
                null_values.append(statistic)
            if sampled == requested:
                break

        _emit(
            progress,
            sampled,
            requested,
            "monte-carlo-sampling",
            int((~is_observed).sum()),
        )

    # Phipson and Smyth plus-one estimator: the observed assignment is always
    # one extremal permutation, even if never selected by the finite RNG draw.
    p_value = (extreme + 1) / (requested + 1)
    variance = p_value * (1.0 - p_value) / (requested + 1)
    standard_error = math.sqrt(max(variance, 0.0))
    margin = 1.96 * standard_error
    ci_low = max(0.0, p_value - margin)
    ci_high = min(1.0, p_value + margin)

    return PermutationResult(
        statistic=observed,
        group1_mean=float(values[:group1_size].mean()),
        group2_mean=float(values[group1_size:].mean()),
        p_value=p_value,
        significant=p_value <= request.alpha,
        alpha=request.alpha,
        method_used="monte_carlo",
        alternative="two-sided",
        total_possible_assignments=total,
        evaluated_permutations=requested,
        extreme_permutations=extreme + 1,
        random_seed=seed,
        standard_error=standard_error,
        confidence_interval_95=(ci_low, ci_high),
        null_distribution=tuple(null_values),
        observed_assignment_excluded=True,
        finite_sample_correction="plus-one:(extreme+1)/(B+1)",
    )


def run_permutation_test(
    request: PermutationRequest,
    progress: ProgressCallback | None = None,
    is_canceled: CancelCallback | None = None,
) -> PermutationResult:
    """Validate and execute the requested two-sided permutation test."""

    values = _check_request(request)
    n = values.size
    total = int(comb(n, request.group1_size, exact=True))

    if request.method == "exact":
        return _exact(
            values,
            request.group1_size,
            request,
            progress,
            is_canceled,
            guarded=True,
        )

    if request.method == "monte_carlo":
        seed = (
            int(np.random.SeedSequence().entropy) & 0xFFFFFFFFFFFFFFFF
            if request.random_seed is None
            else request.random_seed
        )
        return _monte_carlo(
            values, request.group1_size, request, progress, is_canceled, seed
        )

    if total <= request.exact_max_combinations:
        return _exact(
            values,
            request.group1_size,
            request,
            progress,
            is_canceled,
            guarded=False,
        )

    seed = (
        int(np.random.SeedSequence().entropy) & 0xFFFFFFFFFFFFFFFF
        if request.random_seed is None
        else request.random_seed
    )
    return _monte_carlo(
        values, request.group1_size, request, progress, is_canceled, seed
    )


def validate_request(request: PermutationRequest) -> np.ndarray:
    """Validate a request without executing an enumeration or RNG draw."""

    return _check_request(request)


def preflight_request(request: PermutationRequest) -> int:
    """Validate a request and enforce a synchronous exact-enumeration guard."""

    values = validate_request(request)
    total = int(comb(values.size, request.group1_size, exact=True))
    if request.method == "exact" and total > request.exact_limit_cap:
        raise ExactLimitExceededError(
            "exact enumeration exceeds the configured safety limit",
            details={
                "total_possible_assignments": total,
                "limit": request.exact_limit_cap,
            },
        )
    return total
