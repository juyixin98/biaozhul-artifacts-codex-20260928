"""Lag-correlation analysis orchestration."""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np

from ..config import CORRELATION_PRECISION, DETREND_METHOD, MIN_PAIRS, STANDARDIZATION
from ..errors import InvalidLagGridError, InvalidSeriesError
from .matching import match_pairs
from .models import AnalysisResult, LagResult, Series
from .statistics import is_effectively_constant, linear_detrend, pearson_from_residuals


MATCHING_RULE = (
    "sparse_series_anchors_one_pair_per_anchor; "
    "each_dense_sample_assigned_to_nearest_anchor_within_tolerance_and_averaged; "
    "nearest-distance_tie_uses_earlier_anchor"
)
CAUSALITY_NOTICE = (
    "A lag-correlation peak is an associative timing estimate and does not establish causality."
)


def analyze_lag_correlation(
    x: Series,
    y: Series,
    min_lag: float,
    max_lag: float,
    step: float,
    tolerance: float,
    request_id: str | None = None,
) -> AnalysisResult:
    _validate_inputs(x, y, min_lag, max_lag, step, tolerance)
    request_id = request_id or str(uuid.uuid4())

    x_times, x_values = _series_arrays(x)
    y_times, y_values = _series_arrays(y)
    x_interval = _median_interval(x_times)
    y_interval = _median_interval(y_times)

    if x_interval >= y_interval:
        anchor_side = "x"
        anchor_times, anchor_values = x_times, x_values
        dense_times, dense_values = y_times, y_values
        anchor_interval, dense_interval = x_interval, y_interval
    else:
        anchor_side = "y"
        anchor_times, anchor_values = y_times, y_values
        dense_times, dense_values = x_times, x_values
        anchor_interval, dense_interval = y_interval, x_interval

    lag_count = int(round((max_lag - min_lag) / step)) + 1
    lags = min_lag + step * np.arange(lag_count, dtype=np.float64)
    lag_results: list[LagResult] = []
    diagnostics: list[dict[str, Any]] = []

    diagnostics.append(
        {
            "request_id": request_id,
            "event_code": "ANALYSIS_STARTED",
            "anchor_side": anchor_side,
            "anchor_samples": len(anchor_times),
            "dense_samples": len(dense_times),
            "anchor_median_interval": _finite_float(anchor_interval),
            "dense_median_interval": _finite_float(dense_interval),
            "tolerance": tolerance,
            "lag_count": lag_count,
            "minimum_pairs": MIN_PAIRS,
            "status": "started",
            "reason": "series validated; sparse side selected by larger median sample interval",
        }
    )

    for lag in (float(value) for value in lags):
        result = _evaluate_lag(
            lag=lag,
            tolerance=tolerance,
            anchor_side=anchor_side,
            anchor_times=anchor_times,
            anchor_values=anchor_values,
            dense_times=dense_times,
            dense_values=dense_values,
        )
        lag_results.append(result)
        diagnostics.append(_lag_diagnostic(request_id, result))

    valid_results = [result for result in lag_results if result.accepted]
    if valid_results:
        peak = max(
            valid_results,
            key=lambda result: (
                result.correlation if result.correlation is not None else -math.inf,
                -abs(result.lag),
                -result.lag,
            ),
        )
        status = "ok"
        failure_category = None
        reason = f"accepted {len(valid_results)} of {lag_count} lag candidates"
        peak_lag: float | None = peak.lag
        peak_correlation: float | None = peak.correlation
        representative: LagResult = peak
    else:
        status = "unknown"
        peak_lag = None
        peak_correlation = None
        if any(result.status == "INSUFFICIENT_VARIANCE" for result in lag_results):
            failure_category = "INSUFFICIENT_VARIANCE"
            reason = "matched pairs exist but at least one detrended side is constant"
        else:
            failure_category = "INSUFFICIENT_OVERLAP"
            max_pairs = max((result.n_pairs for result in lag_results), default=0)
            reason = f"no lag reached minimum_pairs={MIN_PAIRS}; largest pair count was {max_pairs}"
        representative = _representative_rejected(lag_results)

    diagnostics.append(
        {
            "request_id": request_id,
            "event_code": "ANALYSIS_COMPLETED",
            "status": status,
            "failure_category": failure_category,
            "valid_lag_count": len(valid_results),
            "evaluated_lag_count": lag_count,
            "peak_lag": peak_lag,
            "peak_correlation": peak_correlation,
            "representative_lag": representative.lag,
            "reason": reason,
        }
    )

    return AnalysisResult(
        request_id=request_id,
        status=status,
        failure_category=failure_category,
        reason=reason,
        peak_lag=peak_lag,
        peak_correlation=peak_correlation,
        anchor_side=anchor_side,
        anchor_median_interval=_finite_float(anchor_interval),
        other_median_interval=_finite_float(dense_interval),
        minimum_pairs=MIN_PAIRS,
        detrend=DETREND_METHOD,
        standardization=STANDARDIZATION,
        valid_lag_count=len(valid_results),
        evaluated_lag_count=lag_count,
        best_lag=representative,
        diagnostics=tuple(diagnostics),
    )


def _evaluate_lag(
    lag: float,
    tolerance: float,
    anchor_side: str,
    anchor_times: np.ndarray,
    anchor_values: np.ndarray,
    dense_times: np.ndarray,
    dense_values: np.ndarray,
) -> LagResult:
    pairs, dense_unmatched = match_pairs(
        anchor_times=anchor_times,
        dense_times=dense_times,
        dense_values=dense_values,
        lag=lag,
        tolerance=tolerance,
        anchor_side=anchor_side,
    )
    dense_used = sum(len(pair.paired_indices) for pair in pairs)
    if len(pairs) < MIN_PAIRS:
        return LagResult(
            lag=lag,
            n_pairs=len(pairs),
            correlation=None,
            accepted=False,
            status="NO_MATCH" if not pairs else "INSUFFICIENT_OVERLAP",
            reason=f"matched anchor pairs={len(pairs)} is below minimum_pairs={MIN_PAIRS}",
            pair_count_from_anchor=len(pairs),
            dense_samples_used=dense_used,
            dense_samples_unmatched=dense_unmatched,
            pair_details=pairs,
        )

    anchor_indices = np.array([pair.anchor_index for pair in pairs], dtype=np.int64)
    matched_times = anchor_times[anchor_indices]
    anchor_matched_values = anchor_values[anchor_indices]
    dense_means = np.array(
        [float(np.mean(dense_values[list(pair.paired_indices)])) for pair in pairs],
        dtype=np.float64,
    )

    anchor_residual = linear_detrend(matched_times, anchor_matched_values)
    dense_residual = linear_detrend(matched_times, dense_means)
    if is_effectively_constant(anchor_residual) or is_effectively_constant(dense_residual):
        return LagResult(
            lag=lag,
            n_pairs=len(pairs),
            correlation=None,
            accepted=False,
            status="INSUFFICIENT_VARIANCE",
            reason="a detrended matched sequence has zero sample variance",
            pair_count_from_anchor=len(pairs),
            dense_samples_used=dense_used,
            dense_samples_unmatched=dense_unmatched,
            pair_details=pairs,
        )

    correlation = pearson_from_residuals(anchor_residual, dense_residual)
    if correlation is None:
        return LagResult(
            lag=lag,
            n_pairs=len(pairs),
            correlation=None,
            accepted=False,
            status="INSUFFICIENT_VARIANCE",
            reason="standardization failed because variance was zero or non-finite",
            pair_count_from_anchor=len(pairs),
            dense_samples_used=dense_used,
            dense_samples_unmatched=dense_unmatched,
            pair_details=pairs,
        )

    return LagResult(
        lag=lag,
        n_pairs=len(pairs),
        correlation=round(correlation, CORRELATION_PRECISION),
        accepted=True,
        status="ACCEPTED",
        reason=f"accepted with {len(pairs)} one-per-anchor pairs",
        pair_count_from_anchor=len(pairs),
        dense_samples_used=dense_used,
        dense_samples_unmatched=dense_unmatched,
        pair_details=pairs,
    )


def _lag_diagnostic(request_id: str, result: LagResult) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "event_code": "LAG_EVALUATED",
        "lag": result.lag,
        "n_pairs": result.n_pairs,
        "correlation": result.correlation,
        "accepted": result.accepted,
        "status": result.status,
        "dense_samples_used": result.dense_samples_used,
        "dense_samples_unmatched": result.dense_samples_unmatched,
        "reason": result.reason,
    }


def _representative_rejected(results: list[LagResult]) -> LagResult:
    if not results:  # pragma: no cover - lag grid validation guarantees at least one point
        raise InvalidLagGridError("lag grid must contain at least one point")
    return max(results, key=lambda result: (result.n_pairs, result.dense_samples_used, -abs(result.lag)))


def _validate_inputs(
    x: Series,
    y: Series,
    min_lag: float,
    max_lag: float,
    step: float,
    tolerance: float,
) -> None:
    for label, series in (("x", x), ("y", y)):
        if len(series.times) < 2 or len(series.values) < 2:
            raise InvalidSeriesError(f"{label} must contain at least two samples")
        if len(series.times) != len(series.values):
            raise InvalidSeriesError(f"{label}.times and {label}.values must have equal length")
        if not all(_finite_number(value) for value in series.times):
            raise InvalidSeriesError(f"{label}.times must contain finite numbers")
        if not all(_finite_number(value) for value in series.values):
            raise InvalidSeriesError(f"{label}.values must contain finite numbers")
        if any(
            series.times[index] <= series.times[index - 1]
            for index in range(1, len(series.times))
        ):
            raise InvalidSeriesError(f"{label}.times must be strictly increasing")

    for name, value in (
        ("min_lag", min_lag),
        ("max_lag", max_lag),
        ("step", step),
        ("tolerance", tolerance),
    ):
        if not _finite_number(value):
            raise InvalidLagGridError(f"{name} must be finite")
    if step <= 0:
        raise InvalidLagGridError("step must be positive")
    if tolerance <= 0:
        raise InvalidLagGridError("tolerance must be positive")
    if min_lag > max_lag:
        raise InvalidLagGridError("min_lag must be less than or equal to max_lag")


def _series_arrays(series: Series) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(series.times, dtype=np.float64),
        np.asarray(series.values, dtype=np.float64),
    )


def _median_interval(times: np.ndarray) -> float:
    if len(times) < 2:
        return math.inf
    return float(np.median(np.diff(times)))


def _finite_number(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0
