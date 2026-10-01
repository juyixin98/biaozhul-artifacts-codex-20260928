"""External solver comparison used by diagnostics and tests, not the core path."""

from __future__ import annotations

import numpy as np

from .preprocessing import PreparedData


def sklearn_lasso_coefficients(
    prepared: PreparedData,
    lambdas: np.ndarray,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return sklearn coefficients in original units for explicit lambda values.

    The function reconstructs the same frozen standardized design used by our
    solver. Sparse inputs are materialized only for this optional external
    comparison; the core solver itself remains sparse-aware.
    """

    from sklearn.linear_model import Lasso

    x = prepared.x.toarray() if prepared.is_sparse else prepared.x
    if prepared.is_sparse:
        pass
        # Reconstruct (X - mean) / scale for this optional external check.
        x = x - prepared.x_means / prepared.scales

    rows = []
    for alpha in np.asarray(lambdas, dtype=float).ravel():
        model = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            precompute=False,
            copy_X=True,
            max_iter=max_iter,
            tol=tol,
            positive=False,
        )
        model.fit(x, prepared.y_centered)
        rows.append(model.coef_)
    standardized = np.asarray(rows)
    return standardized / prepared.scales
