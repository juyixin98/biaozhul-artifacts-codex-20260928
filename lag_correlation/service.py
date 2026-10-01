"""Application service mapping contracts to the statistical kernel."""

from __future__ import annotations

from typing import Any

from .config import ALGORITHM_VERSION, SCHEMA_VERSION
from .contracts import AnalysisRequest
from .kernel.analysis import CAUSALITY_NOTICE, MATCHING_RULE, analyze_lag_correlation
from .kernel.models import AnalysisResult, Series


def run_analysis(request: AnalysisRequest) -> AnalysisResult:
    return analyze_lag_correlation(
        x=Series(name=request.x.name, times=request.x.times, values=request.x.values),
        y=Series(name=request.y.name, times=request.y.times, values=request.y.values),
        min_lag=request.lag_grid.min_lag,
        max_lag=request.lag_grid.max_lag,
        step=request.lag_grid.step,
        tolerance=request.matching.tolerance,
        request_id=request.request_id,
    )


def result_to_response(result: AnalysisResult, include_best_pairs: bool = True) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "status": result.status,
        "failure_category": result.failure_category,
        "reason": result.reason,
        "peak_lag": result.peak_lag,
        "peak_correlation": result.peak_correlation,
        "anchor_side": result.anchor_side,
        "valid_lag_count": result.valid_lag_count,
        "evaluated_lag_count": result.evaluated_lag_count,
        "minimum_pairs": result.minimum_pairs,
        "detrend": result.detrend,
        "standardization": result.standardization,
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "matching_rule": MATCHING_RULE,
        "causality_notice": CAUSALITY_NOTICE,
        "best_lag": (
            result.best_lag.to_dict(include_pairs=include_best_pairs)
            if result.best_lag is not None
            else None
        ),
        "diagnostics": list(result.diagnostics),
    }
