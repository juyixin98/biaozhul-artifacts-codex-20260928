from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .errors import InvalidParameterError, InvalidSeriesError, PruningVerificationError
from .models import DpState, Segment, SegmentationResult, segment_cost_key
from .statistics import StablePrefixStats

ProgressCallback = Callable[[str, dict], None]


def _silent_progress(_event: str, _payload: dict) -> None:
    return None


def validate_inputs(values: list[float], min_segment_length: int, penalty: float) -> int:
    if not isinstance(min_segment_length, int) or isinstance(min_segment_length, bool):
        raise InvalidParameterError("min_segment_length must be an integer")
    if min_segment_length < 1:
        raise InvalidParameterError("min_segment_length must be at least 1")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise InvalidParameterError("penalty must be a finite, non-negative number")

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise InvalidSeriesError("values must be a one-dimensional sequence")
    n = int(array.size)
    if n == 0:
        raise InvalidSeriesError("values must contain at least one observation")
    if not np.all(np.isfinite(array)):
        raise InvalidSeriesError("all values must be finite")
    if min_segment_length > n:
        raise InvalidParameterError(
            "min_segment_length cannot exceed the number of observations",
            details={"min_segment_length": min_segment_length, "n": n},
        )
    return n


def _dynamic_programming(
    stats: StablePrefixStats,
    min_segment_length: int,
    penalty: float,
    *,
    prune: bool,
    progress: ProgressCallback,
    run_id: str | None = None,
) -> tuple[SegmentationResult, list[DpState | None]]:
    n = stats.n
    states: list[DpState | None] = [
        DpState(
            objective=0.0,
            sse=0.0,
            previous=-1,
            breakpoints=(),
            segment_count=0,
            transition_count=0,
            pruned_transitions=0,
        )
    ] + [None] * n
    transition_count = 0
    pruned_transitions = 0

    for end in range(1, n + 1):
        best: DpState | None = None
        evaluated = 0
        eligible = 0
        if end < n and n - end < min_segment_length:
            progress(
                "dp_skip_illegal_prefix",
                {
                    "run_id": run_id,
                    "end": end,
                    "reason": "remaining_observations_shorter_than_minimum_segment",
                },
            )
        else:
            starts = range(0, end - min_segment_length + 1)
            for start in starts:
                predecessor = states[start]
                if predecessor is None:
                    continue
                eligible += 1
                # Any continuation adds a non-negative SSE segment plus penalty.
                lower_bound = predecessor.objective + penalty
                if prune and best is not None and lower_bound > best.objective:
                    pruned_transitions += 1
                    transition_count += 1
                    continue

                cost = stats.segment_sse(start, end)
                candidate_sse = predecessor.sse + cost
                candidate_objective = predecessor.objective + cost + penalty
                candidate_breakpoints = predecessor.breakpoints + (end,)
                candidate = DpState(
                    objective=candidate_objective,
                    sse=candidate_sse,
                    previous=start,
                    breakpoints=candidate_breakpoints,
                    segment_count=predecessor.segment_count + 1,
                    transition_count=transition_count + 1,
                    pruned_transitions=pruned_transitions,
                )
                evaluated += 1
                transition_count += 1
                if (
                    best is None
                    or segment_cost_key(
                        candidate.objective, candidate.segment_count, candidate.breakpoints
                    )
                    < segment_cost_key(best.objective, best.segment_count, best.breakpoints)
                ):
                    best = candidate

        states[end] = best
        if best is not None and (end == n or end % max(1, n // 10) == 0):
            progress(
                "dp_progress",
                {
                    "run_id": run_id,
                    "end": end,
                    "eligible_transitions": eligible,
                    "evaluated_transitions": evaluated,
                    "best_segments": best.segment_count,
                    "best_objective": best.objective,
                    "pruning_enabled": prune,
                },
            )

    final = states[n]
    if final is None:
        raise InvalidParameterError(
            "no legal segmentation exists",
            details={"n": n, "min_segment_length": min_segment_length},
        )
    assert final is not None
    result = _rebuild_result(
        stats,
        final,
        penalty=penalty,
        solution_mode="pruned_dp" if prune else "unpruned_dp",
        verification_mode="not_verified",
    )
    return result, states


def _rebuild_result(
    stats: StablePrefixStats,
    final: DpState,
    *,
    penalty: float,
    solution_mode: str,
    verification_mode: str,
) -> SegmentationResult:
    boundaries = (0,) + final.breakpoints
    internal_breakpoints = final.breakpoints[:-1]
    segments: list[Segment] = []
    costs: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mean = stats.segment_mean(start, end)
        cost = stats.segment_sse(start, end)
        segments.append(Segment(start=start, end=end, mean=mean, cost=cost))
        costs.append(cost)

    canonical_sse = math.fsum(segment.cost for segment in segments)
    canonical_total_penalty = float(len(segments)) * penalty
    canonical_objective = math.fsum([canonical_sse, canonical_total_penalty])
    tolerance = 1e-9 * max(
        1.0,
        abs(canonical_objective),
        abs(final.objective),
    )
    if abs(canonical_objective - final.objective) > tolerance:
        raise PruningVerificationError(
            "canonical objective does not match dynamic-programming accumulator",
            details={
                "canonical_objective": canonical_objective,
                "dp_objective": final.objective,
                "tolerance": tolerance,
            },
        )

    return SegmentationResult(
        objective=canonical_objective,
        sse=canonical_sse,
        penalty_per_segment=penalty,
        total_penalty=canonical_total_penalty,
        segment_count=len(segments),
        breakpoints=internal_breakpoints,
        segments=tuple(segments),
        solution_mode=solution_mode,
        transition_count=final.transition_count,
        pruned_transitions=final.pruned_transitions,
        verification_mode=verification_mode,
    )


def solve(
    values: list[float],
    *,
    min_segment_length: int = 1,
    penalty: float = 0.0,
    prune: bool = True,
    verify_pruning: bool = True,
    progress: ProgressCallback | None = None,
    run_id: str | None = None,
) -> SegmentationResult:
    n = validate_inputs(values, min_segment_length, penalty)
    emit = progress or _silent_progress
    stats = StablePrefixStats(values)
    emit(
        "prefix_stats_ready",
        {
            "run_id": run_id,
            "n": n,
            "min_segment_length": min_segment_length,
            "penalty": penalty,
            "prune": prune,
            "verify_pruning": verify_pruning,
        },
    )

    pruned, _ = _dynamic_programming(
        stats,
        min_segment_length,
        penalty,
        prune=prune,
        progress=emit,
        run_id=run_id,
    )

    if prune and verify_pruning:
        emit("pruning_verification_start", {"run_id": run_id})
        unpruned, _ = _dynamic_programming(
            stats,
            min_segment_length,
            penalty,
            prune=False,
            progress=emit,
            run_id=run_id,
        )
        objective_tolerance = 1e-9 * max(1.0, abs(pruned.objective), abs(unpruned.objective))
        if (
            abs(pruned.objective - unpruned.objective) > objective_tolerance
            or pruned.breakpoints != unpruned.breakpoints
            or pruned.segment_count != unpruned.segment_count
        ):
            raise PruningVerificationError(
                "pruned and exhaustive transitions selected different optima",
                details={
                    "pruned_objective": pruned.objective,
                    "unpruned_objective": unpruned.objective,
                    "pruned_breakpoints": pruned.breakpoints,
                    "unpruned_breakpoints": unpruned.breakpoints,
                    "tolerance": objective_tolerance,
                },
            )
        pruned = SegmentationResult(
            objective=pruned.objective,
            sse=pruned.sse,
            penalty_per_segment=pruned.penalty_per_segment,
            total_penalty=pruned.total_penalty,
            segment_count=pruned.segment_count,
            breakpoints=pruned.breakpoints,
            segments=pruned.segments,
            solution_mode="pruned_dp",
            transition_count=pruned.transition_count,
            pruned_transitions=pruned.pruned_transitions,
            verification_mode="matched_unpruned_dp",
        )
        emit(
            "pruning_verification_passed",
            {
                "run_id": run_id,
                "objective_tolerance": objective_tolerance,
                "pruned_transitions": pruned.pruned_transitions,
                "unpruned_transitions": unpruned.transition_count,
            },
        )

    return pruned
