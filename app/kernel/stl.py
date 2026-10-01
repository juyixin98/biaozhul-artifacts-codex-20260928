"""Restricted, additive seasonal-trend decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.config import (
    FIXED_CONFIG,
    INNER_ITERATIONS,
    LOW_PASS_DEGREE,
    LOW_PASS_WINDOW,
    MAX_OBSERVATIONS,
    MIN_OBSERVATIONS,
    MODEL_VERSION,
    OUTER_ROBUST_ITERATIONS,
    PERIOD,
    SEASONAL_DEGREE,
    SEASONAL_WINDOW,
    SPEC_VERSION,
    TREND_DEGREE,
    TREND_WINDOW,
    config_fingerprint,
)
from app.errors import ComputationError, InputError
from app.errors import ResourceExhaustedError
from app.kernel.filters import moving_average
from app.kernel.loess import loess_fit_at, loess_smooth


@dataclass(frozen=True)
class DecompositionResult:
    trend: NDArray[np.float64]
    seasonal: NDArray[np.float64]
    residual: NDArray[np.float64]
    weights: NDArray[np.float64]
    diagnostics: dict[str, Any]
    trace: list[dict[str, Any]]


def _as_clean_series(values: Any) -> NDArray[np.float64]:
    try:
        observed = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError("values must be a one-dimensional numeric sequence") from exc

    if observed.ndim != 1 or observed.size == 0:
        raise InputError(
            "values must be a non-empty one-dimensional sequence",
            {"shape": list(observed.shape)},
        )
    if observed.size < MIN_OBSERVATIONS:
        raise InputError(
            "at least two complete periods are required",
            {
                "received_observations": int(observed.size),
                "required_observations": int(MIN_OBSERVATIONS),
                "period": int(PERIOD),
            },
        )
    if observed.size > MAX_OBSERVATIONS:
        raise ResourceExhaustedError(
            "series exceeds the configured local-processing limit",
            {
                "received_observations": int(observed.size),
                "maximum_observations": int(MAX_OBSERVATIONS),
            },
        )
    missing = np.flatnonzero(~np.isfinite(observed)).astype(int).tolist()
    if missing:
        raise InputError(
            "missing, NaN, or infinite observations are rejected; no zero fill is applied",
            {"non_finite_indices": missing[:20], "non_finite_count": len(missing)},
        )
    return observed


def _array_summary(name: str, values: NDArray[np.float64]) -> dict[str, Any]:
    return {
        "name": name,
        "length": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "first3": values[:3].round(12).tolist(),
        "last3": values[-3:].round(12).tolist(),
    }


def _seasonal_smoother(
    detrended: NDArray[np.float64],
    robust_weights: NDArray[np.float64] | None,
    messages: list[str],
) -> NDArray[np.float64]:
    n = detrended.size
    extended = np.empty(n + 2 * PERIOD, dtype=np.float64)

    for phase in range(PERIOD):
        phase_values = detrended[phase::PERIOD]
        phase_weights = (
            None if robust_weights is None else robust_weights[phase::PERIOD]
        )
        k = phase_values.size

        interior, fallbacks = loess_smooth(
            phase_values, SEASONAL_WINDOW, SEASONAL_DEGREE, phase_weights
        )
        if fallbacks:
            messages.append(
                f"seasonal_phase_{phase}:{fallbacks}_interior_zero_weight_fallbacks"
            )

        extended[PERIOD + phase : n + PERIOD : PERIOD] = interior

        left = loess_fit_at(
            phase_values, -1, SEASONAL_WINDOW, SEASONAL_DEGREE, phase_weights
        )
        if np.isnan(left.value):
            extended[phase] = interior[0]
            messages.append(f"seasonal_phase_{phase}:left_extension_fallback")
        else:
            extended[phase] = left.value

        right = loess_fit_at(
            phase_values, k, SEASONAL_WINDOW, SEASONAL_DEGREE, phase_weights
        )
        right_index = (k + 1) * PERIOD + phase
        if np.isnan(right.value):
            extended[right_index] = interior[-1]
            messages.append(f"seasonal_phase_{phase}:right_extension_fallback")
        else:
            extended[right_index] = right.value

    return extended


def _low_pass(seasonal_extended: NDArray[np.float64]) -> NDArray[np.float64]:
    filtered = moving_average(seasonal_extended, PERIOD)
    filtered = moving_average(filtered, PERIOD)
    filtered = moving_average(filtered, 3)
    loess, _ = loess_smooth(filtered, LOW_PASS_WINDOW, LOW_PASS_DEGREE)
    return loess


def _robust_weights(
    observed: NDArray[np.float64],
    trend: NDArray[np.float64],
    seasonal: NDArray[np.float64],
) -> NDArray[np.float64]:
    absolute = np.abs(observed - trend - seasonal)
    n = observed.size
    sorted_abs = np.sort(absolute)
    # The NETLIB indices are 1-based n/2 and n/2+1 for even n; in zero-based
    # sorting they become n/2-1 and n/2.
    upper_middle = n // 2
    lower_middle = upper_middle - 1 if n % 2 == 0 else upper_middle
    cmad = 3.0 * (sorted_abs[lower_middle] + sorted_abs[upper_middle])
    if cmad == 0.0:
        return np.ones(n, dtype=np.float64)
    lower = 0.001 * cmad
    upper = 0.999 * cmad
    weights = np.ones(n, dtype=np.float64)
    middle = (absolute > lower) & (absolute <= upper)
    weights[middle] = (1.0 - (absolute[middle] / cmad) ** 2) ** 2
    weights[absolute <= lower] = 1.0
    weights[absolute > upper] = 0.0
    return weights


def _record_trace(
    trace: list[dict[str, Any]],
    stage: str,
    outer_index: int,
    inner_index: int,
    observed: NDArray[np.float64],
    trend: NDArray[np.float64],
    seasonal: NDArray[np.float64],
    weights: NDArray[np.float64] | None,
    messages: list[str],
) -> None:
    residual = observed - trend - seasonal
    trace.append(
        {
            "stage": stage,
            "outer_iteration": outer_index,
            "inner_iteration": inner_index,
            "state": {
                "observed": _array_summary("observed", observed),
                "trend": _array_summary("trend", trend),
                "seasonal": _array_summary("seasonal", seasonal),
                "residual": _array_summary("residual", residual),
                "weights": None
                if weights is None
                else _array_summary("weights", weights),
            },
            "judgment": {
                "additive_max_abs_error": float(np.max(np.abs(residual))),
                "seasonal_mean_before_centering": float(np.mean(seasonal)),
                "min_weight": 1.0 if weights is None else float(np.min(weights)),
                "zero_weight_count": 0
                if weights is None
                else int(np.count_nonzero(weights == 0.0)),
            },
            "messages": messages.copy(),
        }
    )


def decompose_stl(values: Any) -> DecompositionResult:
    """Decompose a finite 1-D series with the project's fixed STL specification."""
    observed = _as_clean_series(values)
    n = observed.size
    trend = np.zeros(n, dtype=np.float64)
    seasonal = np.zeros(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    trace: list[dict[str, Any]] = []

    # The initial non-robust pass plus five robust passes matches STL(no=5).
    for outer_index in range(OUTER_ROBUST_ITERATIONS + 1):
        active_weights = weights if outer_index > 0 else None
        for inner_index in range(1, INNER_ITERATIONS + 1):
            messages: list[str] = []
            detrended = observed - trend
            extended_seasonal = _seasonal_smoother(
                detrended, active_weights, messages
            )
            low_pass_trend = _low_pass(extended_seasonal)
            seasonal = extended_seasonal[PERIOD : n + PERIOD] - low_pass_trend
            seasonally_adjusted = observed - seasonal
            trend, fallbacks = loess_smooth(
                seasonally_adjusted,
                TREND_WINDOW,
                TREND_DEGREE,
                active_weights,
            )
            if fallbacks:
                messages.append(f"trend:{fallbacks}_zero_weight_fallbacks")
            _record_trace(
                trace,
                "inner",
                outer_index,
                inner_index,
                observed,
                trend,
                seasonal,
                active_weights,
                messages,
            )

        if outer_index < OUTER_ROBUST_ITERATIONS:
            weights = _robust_weights(observed, trend, seasonal)
            _record_trace(
                trace,
                "robust_weight_update",
                outer_index + 1,
                0,
                observed,
                trend,
                seasonal,
                weights,
                ["biweight_weights_updated_from_residual_median_absolute_deviation"],
            )

    seasonal_shift = float(np.mean(seasonal))
    seasonal = seasonal - seasonal_shift
    trend = trend + seasonal_shift
    residual = observed - trend - seasonal

    reconstruction_error = float(
        np.max(np.abs(observed - (trend + seasonal + residual)))
    )
    if not np.isfinite(trend).all() or not np.isfinite(seasonal).all():
        raise ComputationError(
            "STL produced non-finite components",
            {"non_finite_trend": int(np.count_nonzero(~np.isfinite(trend))),
             "non_finite_seasonal": int(np.count_nonzero(~np.isfinite(seasonal)))},
        )

    diagnostics = {
        "spec_version": SPEC_VERSION,
        "model_version": MODEL_VERSION,
        "config_fingerprint": config_fingerprint(),
        "fixed_config": FIXED_CONFIG,
        "observations": int(n),
        "additive_identity": "observed = trend + seasonal + residual",
        "additive_max_abs_error": reconstruction_error,
        "residual_max_abs": float(np.max(np.abs(residual))),
        "residual_identity_roundtrip_error": reconstruction_error,
        "seasonal_center": float(np.mean(seasonal)),
        "seasonal_shift_applied": seasonal_shift,
        "min_weight": float(np.min(weights)),
        "zero_weight_count": int(np.count_nonzero(weights == 0.0)),
        "weight_formula": "bisquare based on 6*median(abs(residual)); STL cmad=3*two-middle-sum",
    }
    return DecompositionResult(
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        weights=weights,
        diagnostics=diagnostics,
        trace=trace,
    )
