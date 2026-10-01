"""Independent exhaustive oracle for short legal segmentations.

This module deliberately uses recursion and direct Welford SSE rather than the
production DP arrays. It exists to provide an externally readable reference
for tests and small reproducibility experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OracleResult:
    change_points: tuple[int, ...]
    segments: tuple[tuple[int, int], ...]
    costs: tuple[float, ...]
    total_sse: float
    objective: float


def welford_sse(samples: list[float] | tuple[float, ...], start: int, end: int) -> float:
    mean = 0.0
    m2 = 0.0
    for k in range(start, end):
        count = k - start + 1
        delta = float(samples[k]) - mean
        mean += delta / count
        m2 += delta * (float(samples[k]) - mean)
    return max(0.0, m2)


def enumerate_legal_segmentations(
    samples: list[float] | tuple[float, ...],
    penalty: float,
    min_segment_length: int = 1,
) -> list[OracleResult]:
    n = len(samples)
    if n == 0:
        raise ValueError("samples must be non-empty")
    if min_segment_length < 1 or min_segment_length > n:
        raise ValueError("illegal min_segment_length")

    results: list[OracleResult] = []

    def visit(start: int, change_points: list[int], prefix: float) -> None:
        remaining = n - start
        max_end = n if start == 0 else n
        minimum_end = start + min_segment_length
        for end in range(minimum_end, max_end + 1):
            if end < n and n - end < min_segment_length:
                continue
            cost = welford_sse(samples, start, end)
            new_penalty = prefix + cost + float(penalty)
            if end == n:
                boundaries = [0, *change_points, n]
                segment_ranges = tuple(
                    (boundaries[i], boundaries[i + 1])
                    for i in range(len(boundaries) - 1)
                )
                costs = tuple(
                    welford_sse(samples, left, right)
                    for left, right in segment_ranges
                )
                results.append(
                    OracleResult(
                        change_points=tuple(change_points),
                        segments=segment_ranges,
                        costs=costs,
                        total_sse=sum(costs),
                        objective=new_penalty,
                    )
                )
            else:
                visit(end, [*change_points, end], new_penalty)

    visit(0, [], 0.0)
    if not results:
        raise RuntimeError("oracle found no legal segmentation")
    return results


def oracle_optimum(
    samples: list[float] | tuple[float, ...],
    penalty: float,
    min_segment_length: int = 1,
) -> OracleResult:
    candidates = enumerate_legal_segmentations(
        samples, penalty, min_segment_length
    )
    return min(
        candidates,
        key=lambda result: (
            result.objective,
            len(result.segments),
            result.change_points,
        ),
    )
