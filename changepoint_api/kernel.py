"""Offline mean change-point dynamic programming kernels."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .errors import NumericalGuardError, UnsupportedConfigurationError
from .statistics_core import (
    PrefixStats,
    build_prefix_stats,
    segment_sse,
    transition_sse,
    verified_segment,
)
from .versions import KERNEL_VERSION, PRUNED_KERNEL_VERSION, TIE_BREAK_RULE

EventSink = Callable[[str, str, dict[str, Any]], None]


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    mean: float
    cost: float

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class KernelResult:
    algorithm: str
    kernel_version: str
    samples: list[float]
    penalty: float
    min_segment_length: int
    change_points: list[int]
    segments: list[Segment]
    total_sse: float
    objective: float
    dp_objective: float
    prefix_anchor: float
    max_abs_error: float
    max_relative_error: float
    clamped_negative_costs: int
    candidate_evaluations: int
    duration_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    tie_break_rule: str = TIE_BREAK_RULE


def _noop_event(
    level: str, event_type: str, details: dict[str, Any]
) -> None:
    return None


def _validate(
    samples: list[float] | np.ndarray,
    penalty: float,
    min_segment_length: int,
    algorithm: str,
    max_exact_samples: int,
    max_pruned_samples: int,
) -> tuple[np.ndarray, int]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise UnsupportedConfigurationError(
            "only one-dimensional sample sequences are supported"
        )
    n = int(x.size)
    if n == 0:
        raise UnsupportedConfigurationError("samples must not be empty")
    if not np.all(np.isfinite(x)):
        raise UnsupportedConfigurationError("all samples must be finite")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise UnsupportedConfigurationError(
            "penalty must be a finite non-negative number"
        )
    if min_segment_length < 1:
        raise UnsupportedConfigurationError(
            "min_segment_length must be at least 1"
        )
    if min_segment_length > n:
        raise UnsupportedConfigurationError(
            f"min_segment_length={min_segment_length} exceeds n={n}; "
            "no legal segmentation exists"
        )
    limit = (
        max_exact_samples if algorithm == "exact_dp" else max_pruned_samples
    )
    if n > limit:
        raise UnsupportedConfigurationError(
            f"{algorithm} supports at most {limit} samples in this "
            f"configuration, received {n}"
        )
    return x, n


def _choose(
    starts: np.ndarray,
    values: np.ndarray,
    segment_counts: np.ndarray,
) -> int:
    """Select by cost, then more segments, then smaller final boundary."""

    order = np.lexsort((starts, -segment_counts, values))
    return int(starts[int(order[0])])


def _progress_points(total: int) -> set[int]:
    if total <= 10:
        return set(range(1, total + 1))
    return {
        max(1, int(round(total * fraction / 10.0)))
        for fraction in range(1, 11)
    }


def _emit_progress(
    emit: EventSink,
    *,
    endpoint: int,
    iteration: int,
    total: int,
    starts: np.ndarray,
    selected_start: int,
    objective: np.longdouble,
    active_count: int | None,
) -> None:
    details: dict[str, Any] = {
        "endpoint": endpoint,
        "iteration": iteration,
        "total_iterations": total,
        "candidate_count": int(starts.size),
        "selected_start": int(selected_start),
        "best_objective": float(objective),
        "decision_basis": (
            "minimize dp[start] + SSE([start, endpoint)) + penalty"
        ),
    }
    if active_count is not None:
        details["active_candidate_count"] = active_count
    emit("INFO", "dp_progress", details)


def _exact_dp(
    stats: PrefixStats,
    n: int,
    penalty: float,
    min_length: int,
    emit: EventSink,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    infinity = np.longdouble(math.inf)
    dp = np.full(n + 1, infinity, dtype=np.longdouble)
    previous = np.full(n + 1, -1, dtype=np.int64)
    segment_count = np.zeros(n + 1, dtype=np.int64)
    dp[0] = np.longdouble(0.0)
    beta = np.longdouble(penalty)
    evaluations = 0
    total = n - min_length + 1
    checkpoints = _progress_points(total)

    for endpoint in range(min_length, n + 1):
        starts = np.arange(0, endpoint - min_length + 1, dtype=np.int64)
        starts = starts[np.isfinite(dp[starts])]
        values = dp[starts] + transition_sse(stats, starts, endpoint) + beta
        evaluations += int(starts.size)
        selected = _choose(starts, values, segment_count[starts])
        selected_index = int(np.flatnonzero(starts == selected)[0])
        dp[endpoint] = values[selected_index]
        previous[endpoint] = selected
        segment_count[endpoint] = segment_count[selected] + 1

        iteration = endpoint - min_length + 1
        if iteration in checkpoints:
            _emit_progress(
                emit,
                endpoint=endpoint,
                iteration=iteration,
                total=total,
                starts=starts,
                selected_start=selected,
                objective=dp[endpoint],
                active_count=None,
            )

    return dp, previous, segment_count, evaluations, {}


def _pruned_dp(
    stats: PrefixStats,
    n: int,
    penalty: float,
    min_length: int,
    emit: EventSink,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    infinity = np.longdouble(math.inf)
    dp = np.full(n + 1, infinity, dtype=np.longdouble)
    previous = np.full(n + 1, -1, dtype=np.int64)
    segment_count = np.zeros(n + 1, dtype=np.int64)
    dp[0] = np.longdouble(0.0)
    beta = np.longdouble(penalty)
    active = [0]
    evaluations = 0
    proof_evaluations = 0
    removed_total = 0
    total = n - min_length + 1
    checkpoints = _progress_points(total)

    for endpoint in range(min_length, n + 1):
        starts = np.asarray(
            [start for start in active if endpoint - start >= min_length],
            dtype=np.int64,
        )
        values = dp[starts] + transition_sse(stats, starts, endpoint) + beta
        evaluations += int(starts.size)
        selected = _choose(starts, values, segment_count[starts])
        selected_index = int(np.flatnonzero(starts == selected)[0])
        dp[endpoint] = values[selected_index]
        previous[endpoint] = selected
        segment_count[endpoint] = segment_count[selected] + 1

        removed: list[int] = []
        future_available = endpoint < n
        if future_available:
            future = np.arange(endpoint + 1, n + 1, dtype=np.int64)
            selected_future = (
                dp[selected] + segment_sse(stats, selected, future) + beta
            )
            proof_evaluations += int(future.size)
            retained: list[int] = []
            for start in active:
                currently_feasible = endpoint - start >= min_length
                if not currently_feasible or start == selected:
                    retained.append(start)
                    continue
                old_future = (
                    dp[start] + segment_sse(stats, start, future) + beta
                )
                proof_evaluations += int(future.size)
                if bool(np.all(old_future > new_future)):
                    removed.append(start)
                else:
                    retained.append(start)
            removed_total += len(removed)
            active = retained
            active.append(endpoint)

        iteration = endpoint - min_length + 1
        if iteration in checkpoints:
            _emit_progress(
                emit,
                endpoint=endpoint,
                iteration=iteration,
                total=total,
                starts=starts,
                selected_start=selected,
                objective=dp[endpoint],
                active_count=len(active),
            )
        emit(
            "DEBUG",
            "pruned_dp_step",
            {
                "endpoint": endpoint,
                "selected_start": int(selected),
                "evaluated_candidates": int(starts.size),
                "removed_candidates": removed,
                "active_candidates": list(active),
                "pruning_judgement": (
                    "future horizon exhaustively checked; ties retained"
                    if future_available
                    else "no future endpoint remains; pruning unnecessary"
                ),
            },
        )

    diagnostics = {
        "final_active_candidates": list(active),
        "removed_candidate_events": removed_total,
        "proof_evaluations": proof_evaluations,
        "pruning_rule": (
            "remove a candidate only if another candidate is strictly "
            "better at every legal future endpoint"
        ),
    }
    return dp, previous, segment_count, evaluations, diagnostics


def _reconstruct(
    stats: PrefixStats,
    n: int,
    penalty: float,
    previous: np.ndarray,
    dp: np.ndarray,
    algorithm: str,
) -> tuple[list[int], list[Segment], float, float, float, float]:
    ends = [n]
    cursor = n
    while cursor > 0:
        predecessor = int(previous[cursor])
        if predecessor < 0:
            raise NumericalGuardError(
                f"{algorithm} left endpoint {cursor} unreachable"
            )
        if predecessor != 0:
            ends.append(predecessor)
        cursor = predecessor
    ends.reverse()

    segments: list[Segment] = []
    total_sse = 0.0
    max_abs_error = 0.0
    max_relative_error = 0.0
    start = 0
    for end in ends:
        mean, cost, prefix_cost, scipy_cost = verified_segment(
            stats, start, end
        )
        scale = max(1.0, abs(cost), abs(prefix_cost), abs(scipy_cost))
        abs_error = max(
            abs(cost - prefix_cost), abs(cost - scipy_cost)
        )
        max_abs_error = max(max_abs_error, abs_error)
        max_relative_error = max(max_relative_error, abs_error / scale)
        segments.append(Segment(start=start, end=end, mean=mean, cost=cost))
        total_sse += cost
        start = end

    objective = total_sse + penalty * len(segments)
    dp_objective = float(dp[n])
    objective_scale = max(1.0, abs(objective), abs(dp_objective))
    if abs(objective - dp_objective) > 1e-10 * objective_scale:
        raise NumericalGuardError(
            f"reconstructed objective {objective:.17g} does not match DP "
            f"optimum {dp_objective:.17g}"
        )
    change_points = [segment.start for segment in segments[1:]]
    return (
        change_points,
        segments,
        total_sse,
        objective,
        max_abs_error,
        max_relative_error,
    )


def detect_mean_changepoints(
    samples: list[float] | np.ndarray,
    *,
    penalty: float,
    min_segment_length: int = 1,
    algorithm: str = "exact_dp",
    max_exact_samples: int = 2000,
    max_pruned_samples: int = 120,
    emit: EventSink = _noop_event,
) -> KernelResult:
    started = time.perf_counter()
    x, n = _validate(
        samples,
        penalty,
        min_segment_length,
        algorithm,
        max_exact_samples,
        max_pruned_samples,
    )
    stats = build_prefix_stats(x)
    emit(
        "INFO",
        "kernel_start",
        {
            "algorithm": algorithm,
            "n": n,
            "penalty": penalty,
            "min_segment_length": min_segment_length,
            "prefix_strategy": (
                "median-anchored Neumaier sums in numpy.longdouble"
            ),
        },
    )
    if algorithm == "exact_dp":
        dp, previous, _, evaluations, diagnostics = _exact_dp(
            stats, n, penalty, min_segment_length, emit
        )
        kernel_version = KERNEL_VERSION
    elif algorithm == "pruned_dp":
        dp, previous, _, evaluations, diagnostics = _pruned_dp(
            stats, n, penalty, min_segment_length, emit
        )
        kernel_version = PRUNED_KERNEL_VERSION
    else:  # pragma: no cover - protected by Pydantic and validation
        raise UnsupportedConfigurationError(f"unknown algorithm {algorithm}")

    (
        change_points,
        segments,
        total_sse,
        objective,
        max_abs_error,
        max_relative_error,
    ) = _reconstruct(stats, n, penalty, previous, dp, algorithm)
    diagnostics["dp_objective"] = float(dp[n])
    diagnostics["num_segments"] = len(segments)
    return KernelResult(
        algorithm=algorithm,
        kernel_version=kernel_version,
        samples=[float(value) for value in x],
        penalty=float(penalty),
        min_segment_length=int(min_segment_length),
        change_points=change_points,
        segments=segments,
        total_sse=float(total_sse),
        objective=float(objective),
        dp_objective=float(dp[n]),
        prefix_anchor=float(stats.anchor),
        max_abs_error=float(max_abs_error),
        max_relative_error=float(max_relative_error),
        clamped_negative_costs=int(stats.clamped_negative_costs),
        candidate_evaluations=int(evaluations),
        duration_seconds=time.perf_counter() - started,
        diagnostics=diagnostics,
    )
