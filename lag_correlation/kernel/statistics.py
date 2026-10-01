"""Fixed detrending, z-scoring, and Pearson correlation."""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def linear_detrend(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return values.astype(np.float64, copy=True)
    regression = linregress(times.astype(np.float64), values.astype(np.float64))
    fitted = regression.intercept + regression.slope * times.astype(np.float64)
    return values.astype(np.float64) - fitted


def sample_z_score(values: np.ndarray) -> np.ndarray | None:
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0.0 or not np.isfinite(standard_deviation):
        return None
    return (values - np.mean(values)) / standard_deviation


def is_effectively_constant(values: np.ndarray, tolerance: float = 1e-12) -> bool:
    return bool(np.allclose(values, values[0], atol=tolerance, rtol=0.0))


def pearson_from_residuals(left: np.ndarray, right: np.ndarray) -> float | None:
    left_z = sample_z_score(left)
    right_z = sample_z_score(right)
    if left_z is None or right_z is None:
        return None
    correlation = float(np.dot(left_z, right_z) / (len(left_z) - 1))
    if not np.isfinite(correlation):
        return None
    return float(np.clip(correlation, -1.0, 1.0))
