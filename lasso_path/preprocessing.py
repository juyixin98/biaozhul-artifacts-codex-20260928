"""Frozen centering/scaling statistics and scale restoration.

Frozen contract (never re-estimated inside a fit, never mutated by solvers):
  * columns are centered by their training mean ``mean_j``;
  * columns are scaled by their *population* standard deviation (ddof=0)
    ``scale_j = sqrt(mean((X_j - mean_j)^2))``;
  * the intercept is never penalized.

The solver works internally on standardized coefficients ``beta_tilde`` but
keeps its iterate in original scale ``w`` where ``beta_tilde_j = scale_j *
w_j``; ``restore_coef`` / ``restore_intercept`` make the mapping explicit and
independently testable.

All kernels are sparse-safe: they never densify a sparse ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ColumnStats:
    """Frozen preprocessing parameters estimated once from training data."""

    mean: np.ndarray  # (p,) column means
    scale: np.ndarray  # (p,) population std, strictly positive for kept columns
    fit_intercept: bool


def estimate_column_stats(X, *, fit_intercept: bool) -> ColumnStats:
    """Estimate frozen statistics without densifying sparse input."""
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        second = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    else:
        mean = X.mean(axis=0)
        second = np.square(X, dtype=np.float64).mean(axis=0)
    var = np.maximum(second - np.square(mean), 0.0)
    if not fit_intercept:
        # No centering without an intercept; scaling still applies.
        mean = np.zeros_like(mean)
        var = second
    scale = np.sqrt(np.maximum(var, 1e-300))
    return ColumnStats(mean=np.asarray(mean, dtype=np.float64),
                       scale=np.asarray(scale, dtype=np.float64),
                       fit_intercept=fit_intercept)


def standardize_coef(w: np.ndarray, stats: ColumnStats) -> np.ndarray:
    """Map original-scale coefficients to standardized scale."""
    return stats.scale * np.asarray(w, dtype=np.float64)


def restore_coef(beta_tilde: np.ndarray, stats: ColumnStats) -> np.ndarray:
    """Map standardized coefficients back to original scale."""
    return np.asarray(beta_tilde, dtype=np.float64) / stats.scale


def restore_intercept(beta_tilde: np.ndarray, y_mean: float, stats: ColumnStats) -> float:
    """Intercept in original scale given standardized coefficients."""
    w = restore_coef(beta_tilde, stats)
    center = y_mean if stats.fit_intercept else 0.0
    return float(center - stats.mean @ w)


def lambda_max(X, y: np.ndarray, stats: ColumnStats) -> float:
    """Smallest lambda for which the all-zero solution is optimal.

    lambda_max = max_j |X_j^T (y - y_mean)| / (n * scale_j), computed with
    sparse-safe products only.
    """
    n = X.shape[0]
    y_c = y - (y.mean() if stats.fit_intercept else 0.0)
    grad = np.asarray(X.T @ y_c).ravel() / n
    if stats.fit_intercept:
        grad = grad - stats.mean * y_c.mean()
    vals = np.abs(grad) / stats.scale
    return float(vals.max(initial=0.0))
