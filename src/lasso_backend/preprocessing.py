"""Frozen feature centering and scaling used by the Lasso solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


class PreprocessingError(ValueError):
    """Raised when input data cannot be safely standardized."""


@dataclass(frozen=True)
class PreparedData:
    """Centered response and the quantities needed for frozen transforms.

    For dense inputs, ``x_centered`` is materialized. For sparse inputs it
    remains a CSC matrix; residual updates use sparse column slices and sparse
    matrix-vector products instead of calling ``toarray()``.
    """

    x: np.ndarray | sparse.csc_matrix
    y_centered: np.ndarray
    x_means: np.ndarray
    y_mean: float
    scales: np.ndarray
    zero_variance: np.ndarray
    is_sparse: bool

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_features(self) -> int:
        return self.x.shape[1]

    def column_dot(self, j: int) -> np.ndarray:
        """Return column ``j`` without densifying a sparse whole matrix."""
        if self.is_sparse:
            return self.x.getcol(j)
        return self.x[:, j]

    def column_inner(self, j: int) -> float:
        """Squared norm of a column in the sparse-safe path."""
        if self.is_sparse:
            col = self.x.getcol(j)
            return float(col.multiply(col).sum())
        return float(np.dot(self.x[:, j], self.x[:, j]))

    def coordinate_denominator(self, j: int) -> float:
        """Return ``||Z_j||^2 / n`` for the effective centered column."""
        if self.zero_variance[j]:
            return 0.0
        if self.is_sparse:
            col = self.x.getcol(j)
            raw_norm_sq = float(col.multiply(col).sum())
            stored_mean = self.x_means[j] / self.scales[j]
            centered_norm_sq = max(
                raw_norm_sq - self.n_samples * stored_mean**2,
                0.0,
            )
            return centered_norm_sq / self.n_samples
        return self.column_inner(j) / self.n_samples

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(self.x @ vector).ravel()

    def to_original_coefficients(self, standardized_coef: np.ndarray) -> np.ndarray:
        """Map standardized-space coefficients back to original feature units."""
        return np.divide(
            standardized_coef,
            self.scales,
            out=np.zeros_like(standardized_coef, dtype=float),
            where=self.scales > 0,
        )

    def original_intercept(self, standardized_coef: np.ndarray) -> float:
        active_means = self.x_means[self.scales > 0]
        active_scales = self.scales[self.scales > 0]
        active_coef = standardized_coef[self.scales > 0]
        return float(
            self.y_mean - np.sum((active_means / active_scales) * active_coef)
        )


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise PreprocessingError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise PreprocessingError(f"{name} contains NaN or infinite values")
    return array


def prepare_data(
    x: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    *,
    zero_variance_policy: str = "reject",
    variance_tolerance: float = 1e-12,
) -> PreparedData:
    """Validate, center, and scale data once for an entire lambda path.

    Scales use the sample standard deviation. Sparse inputs are converted to
    CSC and never densified; the centered sparse design is represented
    implicitly by a residual offset.
    """

    if zero_variance_policy not in {"reject", "zero_fill"}:
        raise PreprocessingError(
            "zero_variance_policy must be 'reject' or 'zero_fill'"
        )
    if variance_tolerance < 0:
        raise PreprocessingError("variance_tolerance must be non-negative")

    y_array = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(y_array)):
        raise PreprocessingError("y contains NaN or infinite values")

    if sparse.issparse(x):
        x_csc = sparse.csc_matrix(x, dtype=float)
        if x_csc.shape[0] != y_array.size:
            raise PreprocessingError("x and y must contain the same row count")
        if not np.all(np.isfinite(x_csc.data)):
            raise PreprocessingError("x contains NaN or infinite values")

        n_samples, n_features = x_csc.shape
        counts = np.diff(x_csc.indptr)
        sums = np.asarray(x_csc.sum(axis=0)).ravel()
        sumsq = np.asarray(x_csc.multiply(x_csc).sum(axis=0)).ravel()
        x_means = sums / n_samples
        centered_sumsq = np.maximum(
            sumsq - sums * sums / n_samples, 0.0
        )
        scales = np.sqrt(centered_sumsq / (n_samples - 1))
    else:
        x_dense = _as_2d_float("x", x)
        if x_dense.shape[0] != y_array.size:
            raise PreprocessingError("x and y must contain the same row count")

        n_samples, n_features = x_dense.shape
        x_means = x_dense.mean(axis=0)
        scales = x_dense.std(axis=0, ddof=1)
        x_csc = x_dense - x_means

    if n_samples < 2:
        raise PreprocessingError("at least two samples are required")

    zero_variance = scales <= variance_tolerance
    if np.any(zero_variance) and zero_variance_policy == "reject":
        indices = np.flatnonzero(zero_variance).tolist()
        shown = indices[:20]
        suffix = "..." if len(indices) > len(shown) else ""
        raise PreprocessingError(
            f"constant or near-constant columns rejected: {shown}{suffix}"
        )

    safe_scales = scales.copy()
    safe_scales[zero_variance] = 1.0
    y_mean = float(y_array.mean())
    y_centered = y_array - y_mean

    if sparse.issparse(x):
        # X/s without subtracting mean. Residual centering below supplies the
        # mean term, while sparse structure is preserved for coordinate work.
        x_scaled = x_csc.multiply(1.0 / safe_scales).tocsr().tocsc()
        zero_cols = np.flatnonzero(zero_variance)
        if zero_cols.size:
            # Ensure filled constant columns contain no explicit entries.
            for column_index in zero_cols:
                start = x_scaled.indptr[column_index]
                end = x_scaled.indptr[column_index + 1]
                x_scaled.data[start:end] = 0.0
        x_scaled.eliminate_zeros()
    else:
        x_scaled = x_csc / safe_scales
        x_scaled[:, zero_variance] = 0.0

    return PreparedData(
        x=x_scaled,
        y_centered=y_centered,
        x_means=x_means,
        y_mean=y_mean,
        scales=safe_scales,
        zero_variance=zero_variance,
        is_sparse=bool(sparse.issparse(x_scaled)),
    )
