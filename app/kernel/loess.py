"""Local polynomial smoothing used by the restricted STL kernel.

This implementation follows the NETLIB STL smoothing rules, including its
tricube weights and one-period seasonal boundary extensions. Jumps are fixed
to one, so no interpolation between fitted points is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.errors import ComputationError


@dataclass(frozen=True)
class LoessFit:
    value: float
    weight_sum: float


def _tricube(distance: float, radius: float) -> float:
    if radius <= 0.0:
        return 1.0
    ratio = abs(distance) / radius
    if ratio <= 0.001:
        return 1.0
    if ratio >= 0.999:
        return 0.0
    return float((1.0 - ratio**3) ** 3)


def _nearest_window(target: int, length: int, n: int) -> tuple[int, int, float]:
    """Return the zero-based STL local window and tricube radius."""
    if length >= n:
        left, right = 0, n - 1
        radius = max(target - left, right - target) + (length - n) // 2
        return left, right, float(radius)

    half = (length + 2) // 2
    if target < half:
        left, right = 0, length - 1
    elif target >= n - half + 1:
        left, right = n - length, n - 1
    else:
        left = target - half + 1
        right = left + length - 1
    radius = max(target - left, right - target)
    return left, right, float(radius)


def loess_fit_at(
    values: NDArray[np.float64],
    target: int,
    length: int,
    degree: int,
    robust_weights: NDArray[np.float64] | None = None,
) -> LoessFit:
    """Fit weighted local polynomial and predict at one integer target."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0 or length <= 0 or degree not in (0, 1):
        raise ComputationError(
            "LOESS received an invalid window or degree",
            {"n": n, "window": length, "degree": degree},
        )

    left, right, radius = _nearest_window(target, length, n)
    points = np.arange(left, right + 1, dtype=np.float64)
    weights = np.array([_tricube(p - target, radius) for p in points])
    if robust_weights is not None:
        weights *= robust_weights[left : right + 1]

    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return LoessFit(value=np.nan, weight_sum=0.0)

    normalized = weights / weight_sum
    equivalent = normalized.copy()

    if degree == 1 and radius > 0.0:
        one_based = points + 1.0
        weighted_mean_x = float(np.dot(normalized, one_based))
        spread = float(np.dot(normalized, (one_based - weighted_mean_x) ** 2))
        if np.sqrt(spread) > 0.001 * (n - 1):
            slope_factor = (float(target + 1) - weighted_mean_x) / spread
            equivalent *= 1.0 + slope_factor * (one_based - weighted_mean_x)

    value = float(np.dot(equivalent, x[left : right + 1]))
    if not np.isfinite(value):
        return LoessFit(value=np.nan, weight_sum=weight_sum)
    return LoessFit(value=value, weight_sum=weight_sum)


def loess_smooth(
    values: NDArray[np.float64],
    length: int,
    degree: int,
    robust_weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Smooth every observed integer location, using STL's zero-weight fallback."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 1:
        return x.copy(), 0

    output = np.empty(n, dtype=np.float64)
    fallbacks = 0
    for target in range(n):
        fit = loess_fit_at(x, target, length, degree, robust_weights)
        if np.isnan(fit.value):
            output[target] = x[target]
            fallbacks += 1
        else:
            output[target] = fit.value
    return output, fallbacks

