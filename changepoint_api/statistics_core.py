"""Numerically stable prefix statistics and segment cost primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import norm

from .errors import NumericalGuardError


@dataclass
class PrefixStats:
    samples: np.ndarray
    anchor: np.longdouble
    sums: np.ndarray
    squared_sums: np.ndarray
    clamped_negative_costs: int = 0


def compensated_cumsum(values: np.ndarray) -> np.ndarray:
    """Neumaier-compensated cumulative sums in extended precision.

    The correction is materialized at every prefix. A median anchor keeps the
    accumulated deviations bounded even when an outlier is present.
    """

    totals = np.zeros(values.size + 1, dtype=np.longdouble)
    corrections = np.zeros_like(totals)
    compensation = np.longdouble(0.0)
    for index, value in enumerate(values):
        adjusted = np.longdouble(value) - compensation
        new_total = totals[index] + adjusted
        compensation = (new_total - totals[index]) - adjusted
        totals[index + 1] = new_total
        corrections[index + 1] = compensation
    return totals - corrections


def build_prefix_stats(samples: list[float] | np.ndarray) -> PrefixStats:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must contain only finite values")

    # Median is used rather than a fixed zero: it is local to the input and
    # prevents extreme magnitudes from dominating every prefix calculation.
    anchor = np.longdouble(np.median(x))
    deviations = x.astype(np.longdouble) - anchor
    sums = compensated_cumsum(deviations)
    squared_sums = compensated_cumsum(deviations * deviations)
    return PrefixStats(
        samples=x,
        anchor=anchor,
        sums=sums,
        squared_sums=squared_sums,
    )


def _guard_and_clamp_sse(
    raw: np.ndarray,
    *,
    lengths: np.ndarray,
    squares: np.ndarray,
    mean_terms: np.ndarray,
) -> np.ndarray:
    scale = np.maximum(
        np.longdouble(1.0),
        np.maximum(np.abs(squares), np.abs(mean_terms)),
    )
    tolerance = np.finfo(np.longdouble).eps * scale * np.longdouble(8.0)
    singleton = lengths == np.longdouble(1.0)
    invalid = (raw < -tolerance) & ~singleton
    if np.any(invalid):
        worst = int(np.argmin(raw))
        raise NumericalGuardError(
            "segment variance was negative beyond extended-precision "
            f"tolerance: index={worst}, sse={float(raw[worst]):.17g}"
        )
    result = np.maximum(raw, np.longdouble(0.0))
    singleton = lengths == np.longdouble(1.0)
    result[singleton] = np.longdouble(0.0)
    return result


def segment_sse(
    stats: PrefixStats,
    start: int,
    ends: int | np.ndarray,
    *,
    record_correction: bool = True,
) -> np.ndarray:
    """Return SSE for half-open segments [start, end)."""

    end_array = np.asarray(ends, dtype=np.int64)
    lengths = (end_array - start).astype(np.longdouble)
    sums = stats.sums[end_array] - stats.sums[start]
    squares = (
        stats.squared_sums[end_array] - stats.squared_sums[start]
    )
    mean_terms = sums * sums / lengths
    raw = squares - mean_terms
    result = _guard_and_clamp_sse(
        raw,
        lengths=lengths,
        squares=squares,
        mean_terms=mean_terms
    )
    singleton = lengths == np.longdouble(1.0)
    negative = (raw < np.longdouble(0.0)) & ~singleton
    if record_correction:
        stats.clamped_negative_costs += int(np.count_nonzero(negative))
    return result


def transition_sse(
    stats: PrefixStats, starts: np.ndarray, endpoint: int
) -> np.ndarray:
    """Vectorized SSE for candidate last segments ending at one endpoint."""

    start_array = np.asarray(starts, dtype=np.int64)
    lengths = (endpoint - start_array).astype(np.longdouble)
    sums = stats.sums[endpoint] - stats.sums[start_array]
    squares = (
        stats.squared_sums[endpoint] - stats.squared_sums[start_array]
    )
    mean_terms = sums * sums / lengths
    raw = squares - mean_terms
    result = _guard_and_clamp_sse(
        raw,
        lengths=lengths,
        squares=squares,
        mean_terms=mean_terms
    )
    singleton = lengths == np.longdouble(1.0)
    result[singleton] = np.longdouble(0.0)
    stats.clamped_negative_costs += int(np.count_nonzero(negative))
    return result


def welford_summary(x: np.ndarray) -> tuple[float, float]:
    """Stable one-pass mean and sum of squared errors."""

    mean = 0.0
    m2 = 0.0
    for count, value in enumerate(x, start=1):
        delta = float(value) - mean
        mean += delta / count
        delta_after = float(value) - mean
        m2 += delta * delta_after
    if m2 < 0.0:
        m2 = 0.0
    return mean, m2


def verified_segment(
    stats: PrefixStats, start: int, end: int
) -> tuple[float, float, float, float]:
    """Return mean, reported SSE, prefix SSE, and SciPy norm SSE."""

    x = stats.samples[start:end]
    mean, welford_sse = welford_summary(x)
    prefix_sse = float(segment_sse(stats, start, np.asarray([end]))[0])
    scipy_sse = float(norm(x - mean)) ** 2
    scale = max(1.0, abs(welford_sse), abs(prefix_sse), abs(scipy_sse))
    max_error = max(
        abs(welford_sse - prefix_sse), abs(welford_sse - scipy_sse)
    )
    if max_error > 1e-11 * scale:
        raise NumericalGuardError(
            f"final SSE checks disagree for [{start},{end}): "
            f"prefix={prefix_sse:.17g}, welford={welford_sse:.17g}, "
            f"scipy={scipy_sse:.17g}"
        )
    return mean, welford_sse, prefix_sse, scipy_sse
