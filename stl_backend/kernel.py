"""Pure NumPy implementation of the constrained STL procedure.

The smoother follows the STL inner loop described by Cleveland et al.:
periodic sub-series LOESS, one-period extension on both boundaries, a
moving-average low-pass filter, centering of the seasonal component, and a
final trend LOESS.  Boundary estimates therefore use explicit LOESS
extrapolation instead of reflection, zero padding, or NaN filling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import StlConfig
from .errors import CalculationError, InputContractError


def _as_float_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputContractError("values must be numeric") from exc
    if y.ndim != 1:
        raise InputContractError("values must be a one-dimensional sequence")
    if y.size == 0:
        raise InputContractError("values must not be empty")
    if not np.all(np.isfinite(y)):
        raise InputContractError(
            "missing, NaN and infinite observations are not accepted; "
            "they must be handled explicitly by the caller"
        )
    return y


def validate_series(values: Sequence[float] | np.ndarray, config: StlConfig) -> np.ndarray:
    """Validate the numeric sample contract used by the kernel."""

    y = _as_float_vector(values)
    if y.size < config.minimum_observations:
        raise InputContractError(
            "at least two complete periods are required",
            details={
                "observations": int(y.size),
                "minimum_observations": config.minimum_observations,
                "period": config.period,
            },
        )
    if y.size > config.max_observations:
        # The API owns the public limit, but direct callers get the same guard.
        from .errors import ResourceExhaustedError

        raise ResourceExhaustedError(
            "requested series exceeds the configured observation limit",
            details={
                "observations": int(y.size),
                "maximum_observations": config.max_observations,
            },
        )
    return y


def _loess_estimate(
    y: np.ndarray,
    window: int,
    degree: int,
    x: float,
    left: int,
    right: int,
    robust_weights: np.ndarray | None = None,
) -> float:
    """Estimate at one-based x over inclusive one-based [left, right]."""

    n = y.size
    xj = np.arange(left, right + 1, dtype=np.float64)
    idx = xj.astype(np.intp) - 1
    h = max(x - left, right - x)
    if window > n:
        h += (window - n) // 2

    distance = np.abs(xj - x)
    near_inner = distance <= 0.001 * h
    within = distance <= 0.999 * h
    weights = np.zeros_like(distance)
    weights[within] = (1.0 - (distance[within] / h) ** 3) ** 3
    weights[near_inner] = 1.0
    if robust_weights is not None:
        weights *= robust_weights[idx]

    total = float(weights.sum())
    if total <= 0.0:
        return np.nan
    weights /= total

    if h > 0.0 and degree > 0:
        x_mean = float(np.dot(weights, xj))
        centered_x = xj - x_mean
        scale = float(np.sqrt(np.dot(weights, centered_x**2)))
        if scale > 0.001 * (n - 1):
            slope_adjustment = (x - x_mean) / (scale**2)
            weights *= slope_adjustment * centered_x + 1.0

    return float(np.dot(weights, y[idx]))


def loess_smooth(
    y: np.ndarray,
    window: int,
    degree: int = 1,
    robust_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Local linear LOESS with fixed nearest-neighbour boundary windows."""

    n = y.size
    if n < 2:
        return y.copy()

    output = np.empty(n, dtype=np.float64)
    if window >= n:
        for i in range(n):
            value = _loess_estimate(
                y, window, degree, float(i + 1), 1, n, robust_weights
            )
            output[i] = value if np.isfinite(value) else y[i]
        return output

    nshift = (window + 2) // 2
    left = 1
    right = window
    for i in range(n):
        if i + 1 > nshift and right != n:
            left += 1
            right += 1
        value = _loess_estimate(
            y, window, degree, float(i + 1), left, right, robust_weights
        )
        output[i] = value if np.isfinite(value) else y[i]
    return output


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window > x.size:
        raise CalculationError("moving-average window exceeds available data")
    return np.convolve(x, np.ones(window, dtype=np.float64) / window, mode="valid")


def seasonal_smooth(
    detrended: np.ndarray,
    robust_weights: np.ndarray | None,
    period: int,
    seasonal_window: int,
    seasonal_degree: int,
) -> np.ndarray:
    """Smooth each seasonal sub-series and extrapolate one period at each end."""

    n = detrended.size
    extended = np.zeros(n + 2 * period, dtype=np.float64)

    for phase in range(period):
        subseries = detrended[phase::period]
        k = subseries.size
        sub_weights = None
        if robust_weights is not None:
            sub_weights = robust_weights[phase::period]

        interior = loess_smooth(
            subseries, seasonal_window, seasonal_degree, sub_weights
        )
        extended[period + phase : period + n : period] = interior

        right_boundary = min(seasonal_window, k)
        left_value = _loess_estimate(
            subseries,
            seasonal_window,
            seasonal_degree,
            0.0,
            1,
            right_boundary,
            sub_weights,
        )
        extended[phase] = left_value if np.isfinite(left_value) else interior[0]

        left_boundary = max(1, k - seasonal_window + 1)
        right_value = _loess_estimate(
            subseries,
            seasonal_window,
            seasonal_degree,
            float(k + 1),
            left_boundary,
            k,
            sub_weights,
        )
        extended[(k + 1) * period + phase] = (
            right_value if np.isfinite(right_value) else interior[-1]
        )

    return extended


def low_pass_smooth(
    extended_season: np.ndarray,
    number_observations: int,
    period: int,
    low_pass_window: int,
    low_pass_degree: int,
) -> np.ndarray:
    """STL filter: period MA, period MA, three-point MA, then LOESS."""

    once = _moving_average(extended_season, period)
    twice = _moving_average(once, period)
    three_point = _moving_average(twice, 3)
    return loess_smooth(three_point, low_pass_window, low_pass_degree)[
        :number_observations
    ]


def robust_bisquare_weights(residual: np.ndarray) -> tuple[np.ndarray, float]:
    """Return STL bi-square weights and the 6*MAD scale used in the cutoff."""

    absolute = np.abs(residual)
    midpoint_0 = absolute.size // 2
    midpoint_1 = absolute.size - midpoint_0 - 1
    ordered = np.partition(absolute, [midpoint_0, midpoint_1])
    cutoff = 3.0 * (ordered[midpoint_0] + ordered[midpoint_1])
    if cutoff == 0.0:
        return np.ones_like(residual), 0.0

    weights = np.ones_like(residual)
    middle = (absolute > 0.001 * cutoff) & (absolute <= 0.999 * cutoff)
    weights[middle] = (1.0 - (absolute[middle] / cutoff) ** 2) ** 2
    weights[absolute > 0.999 * cutoff] = 0.0
    return weights, float(cutoff)


def stl_decompose(
    values: Sequence[float] | np.ndarray,
    config: StlConfig,
    *,
    collect_history: bool = True,
) -> dict[str, Any]:
    """Run the fixed constrained STL decomposition.

    Returns arrays plus explicit iteration diagnostics. Missing values are an
    error here; they are never implicitly converted to zero.
    """

    y = validate_series(values, config)
    n = y.size
    trend = np.zeros(n, dtype=np.float64)
    season = np.zeros(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    history: list[dict[str, Any]] = []

    try:
        for outer in range(config.outer_iterations + 1):
            use_weights = outer > 0
            for _ in range(config.inner_iterations):
                detrended = y - trend
                extended_season = seasonal_smooth(
                    detrended,
                    weights if use_weights else None,
                    config.period,
                    config.seasonal_window,
                    config.seasonal_degree,
                )
                low_pass = low_pass_smooth(
                    extended_season,
                    n,
                    config.period,
                    config.low_pass_window,
                    config.low_pass_degree,
                )
                season = (
                    extended_season[config.period : config.period + n] - low_pass
                )
                trend = loess_smooth(
                    y - season,
                    config.trend_window,
                    config.trend_degree,
                    weights if use_weights else None,
                )

            residual = y - season - trend
            reconstruction_error = float(np.max(np.abs(y - (trend + season + residual))))
            if collect_history:
                history.append(
                    {
                        "outer_iteration": outer,
                        "robust_weights_applied": use_weights,
                        "max_abs_residual": float(np.max(np.abs(residual))),
                        "mean_abs_residual": float(np.mean(np.abs(residual))),
                        "rms_residual": float(np.sqrt(np.mean(residual**2))),
                        "min_weight_used": float(np.min(weights)),
                        "zero_weights_used": int(np.sum(weights == 0.0)),
                        "reconstruction_max_abs_error": reconstruction_error,
                        "season_mean_before_final_centering": float(
                            np.mean(season)
                        ),
                    }
                )

            if outer == config.outer_iterations:
                break
            weights, scale_cutoff = robust_bisquare_weights(residual)
            if collect_history:
                history[-1]["robust_scale_cutoff"] = scale_cutoff
                history[-1]["min_weight_for_next_iteration"] = float(
                    np.min(weights)
                )
                history[-1]["zero_weights_for_next_iteration"] = int(
                    np.sum(weights == 0.0)
                )

        centering_shift = float(np.mean(season))
        # Allocate the identifiability constant to the trend, matching the
        # usual centered-STL convention while preserving additive identity.
        season = season - centering_shift
        trend = trend + centering_shift
        residual = y - trend - season

        if not (
            np.all(np.isfinite(trend))
            and np.all(np.isfinite(season))
            and np.all(np.isfinite(residual))
            and np.all(np.isfinite(weights))
        ):
            raise CalculationError("decomposition produced a non-finite result")

        return {
            "values": y,
            "trend": trend,
            "seasonal": season,
            "residual": residual,
            "robust_weights": weights,
            "history": history,
            "centering_shift": centering_shift,
            "kernel_version": config.kernel_version,
        }
    except CalculationError:
        raise
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise CalculationError("STL calculation failed") from exc
