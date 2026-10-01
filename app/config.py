"""Fixed STL specification and service limits."""

from __future__ import annotations

import hashlib
import json
from typing import Any

MODEL_VERSION = "restricted-stl-additive-v1"
SPEC_VERSION = "1.0.0"

PERIOD = 7
SEASONAL_WINDOW = 7
TREND_WINDOW = 15
LOW_PASS_WINDOW = 9
SEASONAL_DEGREE = 1
TREND_DEGREE = 1
LOW_PASS_DEGREE = 1
INNER_ITERATIONS = 2
OUTER_ROBUST_ITERATIONS = 5
BOUNDARY_STRATEGY = (
    "local_linear_loess_at_observed_points;"
    "seasonal_subseries_extends_one_period_at_each_end"
)

MIN_OBSERVATIONS = 2 * PERIOD
MAX_OBSERVATIONS = 5_000

FIXED_CONFIG: dict[str, Any] = {
    "model": "additive",
    "period": PERIOD,
    "seasonal_window": SEASONAL_WINDOW,
    "trend_window": TREND_WINDOW,
    "low_pass_window": LOW_PASS_WINDOW,
    "seasonal_degree": SEASONAL_DEGREE,
    "trend_degree": TREND_DEGREE,
    "low_pass_degree": LOW_PASS_DEGREE,
    "inner_iterations": INNER_ITERATIONS,
    "outer_robust_iterations": OUTER_ROBUST_ITERATIONS,
    "jump": 1,
    "boundary_strategy": BOUNDARY_STRATEGY,
    "seasonal_centering": "subtract_sample_mean_after_final_iteration",
    "missing_value_policy": "reject_non_finite_values;_no_zero_fill",
    "minimum_observations": MIN_OBSERVATIONS,
}


def config_fingerprint() -> str:
    """Return a stable fingerprint of the frozen mathematical configuration."""
    payload = json.dumps(FIXED_CONFIG, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

