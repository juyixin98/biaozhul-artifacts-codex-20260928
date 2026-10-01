from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from numbers import Real

import numpy as np
from scipy.linalg import norm

from .errors import InvalidParameterError, InvalidSeriesError


@dataclass(frozen=True)
class ReferenceSegment:
    start: int
    end: int
    mean: Fraction | float
    cost: Fraction | float


@dataclass(frozen=True)
class ReferenceSolution:
    objective: Fraction | float
    sse: Fraction | float
    breakpoints: tuple[int, ...]
    segments: tuple[ReferenceSegment, ...]
    considered_segmentations: int
    exact: bool


def _as_rational(values: list[float]) -> list[Fraction] | None:
    result: list[Fraction] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        result.append(Fraction(value, 1))
    return result


def _legal_cut_sets(n: int, min_segment_length: int) -> list[tuple[int, ...]]:
    internal = range(1, n)
    cuts = []
    max_segments = n // min_segment_length
    for segment_count in range(1, max_segments + 1):
        for chosen in combinations(internal, segment_count - 1):
            boundaries = (0,) + chosen + (n,)
            if all(
                boundaries[index + 1] - boundaries[index] >= min_segment_length
                for index in range(len(boundaries) - 1)
            ):
                cuts.append(chosen)
    return cuts


def enumerate_best_segmentation(
    values: list[float] | list[int],
    min_segment_length: int,
    penalty: float | Fraction,
) -> ReferenceSolution:
    """Brute-force reference intended only for short test series.

    Integer fixtures use Fraction arithmetic, giving an exact oracle. Floating
    fixtures independently use a SciPy vector norm rather than the production
    prefix statistics.
    """

    if not isinstance(min_segment_length, int) or min_segment_length < 1:
        raise InvalidParameterError("min_segment_length must be a positive integer")
    if len(values) == 0:
        raise InvalidSeriesError("reference requires at least one observation")
    n = len(values)
    if min_segment_length > n:
        raise InvalidParameterError("min_segment_length cannot exceed n")

    values_list = list(values)
    if not all(isinstance(value, Real) and math.isfinite(float(value)) for value in values_list):
        raise InvalidSeriesError("all reference values must be finite")
    rational_values = _as_rational(values_list)
    exact = rational_values is not None
    penalty_float = float(penalty)
    if not math.isfinite(penalty_float) or penalty_float < 0.0:
        raise InvalidParameterError("penalty must be finite and non-negative")
    penalty_value: Fraction | float = Fraction(penalty_float) if exact else penalty_float

    best: ReferenceSolution | None = None
    considered = 0
    for cuts in _legal_cut_sets(n, min_segment_length):
        boundaries = (0,) + cuts + (n,)
        segments: list[ReferenceSegment] = []
        costs: list[Fraction | float] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if exact:
                data = rational_values[start:end]  # type: ignore[index]
                mean = sum(data, Fraction(0, 1)) / (end - start)
                cost = sum((x - mean) ** 2 for x in data)
            else:
                data = np.asarray(values[start:end], dtype=np.float64)
                if not np.all(np.isfinite(data)):
                    raise InvalidSeriesError("all reference values must be finite")
                mean = math.fsum(float(x) for x in data) / float(data.size)
                cost = float(norm(data - mean)) ** 2
            segments.append(ReferenceSegment(start, end, mean, cost))
            costs.append(cost)

        segment_count = len(segments)
        if exact:
            sse = sum(costs, Fraction(0, 1))
            objective = sse + segment_count * penalty_value
        else:
            sse = math.fsum(float(cost) for cost in costs)
            objective = math.fsum([sse, float(segment_count) * float(penalty_value)])
        solution = ReferenceSolution(
            objective=objective,
            sse=sse,
            breakpoints=tuple(int(c) for c in cuts),
            segments=tuple(segments),
            considered_segmentations=0,
            exact=exact,
        )
        considered += 1
        if best is None or (
            objective,
            segment_count,
            solution.breakpoints,
        ) < (
            best.objective,
            len(best.segments),
            best.breakpoints,
        ):
            best = solution

    assert best is not None
    return ReferenceSolution(
        objective=best.objective,
        sse=best.sse,
        breakpoints=best.breakpoints,
        segments=best.segments,
        considered_segmentations=considered,
        exact=best.exact,
    )
