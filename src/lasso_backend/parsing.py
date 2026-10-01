"""Convert API payloads into validated NumPy/SciPy arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .config import Settings
from .errors import LassoRequestError
from .models import LassoPathRequest


def _validate_shape(n_rows: int, n_cols: int, nnz: int, settings: Settings) -> None:
    if n_rows > settings.max_rows:
        raise LassoRequestError(
            f"row count {n_rows} exceeds limit {settings.max_rows}"
        )
    if n_cols > settings.max_features:
        raise LassoRequestError(
            f"feature count {n_cols} exceeds limit {settings.max_features}"
        )
    if nnz > settings.max_nnz:
        raise LassoRequestError(
            f"non-zero entry count {nnz} exceeds limit {settings.max_nnz}"
        )


def parse_matrix(payload: dict[str, Any] | list[list[float]], settings: Settings):
    if isinstance(payload, list):
        values = np.asarray(payload, dtype=float)
        _validate_shape(values.shape[0], values.shape[1], values.size, settings)
        return values

    matrix_format = payload.get("format")
    if matrix_format == "dense":
        values = np.asarray(payload.get("values"), dtype=float)
        if values.ndim != 2:
            raise LassoRequestError("dense matrix values must be two-dimensional")
        _validate_shape(values.shape[0], values.shape[1], values.size, settings)
        return values

    if matrix_format != "coo":
        raise LassoRequestError(
            "x.format must be 'coo' for sparse data or 'dense'"
        )

    rows = np.asarray(payload.get("rows", []), dtype=int)
    cols = np.asarray(payload.get("cols", []), dtype=int)
    data = np.asarray(payload.get("data", []), dtype=float)
    n_rows = int(payload.get("n_rows", 0))
    n_cols = int(payload.get("n_cols", 0))
    if n_rows <= 0 or n_cols <= 0:
        raise LassoRequestError("sparse n_rows and n_cols must be positive")
    _validate_shape(n_rows, n_cols, int(rows.size), settings)
    if rows.size:
        linear_indices = rows * n_cols + cols
        if np.unique(linear_indices).size != rows.size:
            raise LassoRequestError(
                "rows/cols contains duplicate matrix coordinates"
            )

    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_rows, n_cols)
    ).tocsc()


def parse_request(
    request: LassoPathRequest,
    settings: Settings,
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray]:
    x = parse_matrix(request.x, settings)
    y = np.asarray(request.y, dtype=float)

    if y.ndim != 1:
        raise LassoRequestError("y must be a one-dimensional array")
    if x.shape[0] != y.size:
        raise LassoRequestError("x and y must contain the same row count")
    return x, y
