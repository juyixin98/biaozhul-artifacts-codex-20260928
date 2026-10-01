"""Solver output vs. closed-form reference on an orthogonal design.

The reference answer is computed in the test from the analytic soft-threshold
formula for orthonormal designs, not by the solver under test.
"""

import numpy as np

from lasso_path.preprocessing import estimate_column_stats
from lasso_path.solver import solve_lasso


def _closed_form(X, y, lam):
    n = X.shape[0]
    y_c = y - y.mean()
    z = X.T @ y_c / n  # columns are centered with ||X_j||^2/n = 1
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def test_matches_closed_form(orthogonal_design):
    X, y, _ = orthogonal_design
    stats = estimate_column_stats(X, fit_intercept=True)
    # Design is already standardized: scales must be ~1, means ~0.
    assert np.allclose(stats.mean, 0.0, atol=1e-12)
    assert np.allclose(stats.scale, 1.0, atol=1e-12)
    for lam in (0.5, 0.2, 0.05):
        res = solve_lasso(X, y, stats, lam, tol=1e-10, max_iter=10000)
        assert res.converged
        expected = _closed_form(X, y, lam)
        assert np.allclose(res.coef, expected, atol=1e-5), (
            f"lambda={lam}: {res.coef} != {expected}")
        # Columns are centered, so the intercept is just y_mean.
        assert abs(res.intercept - y.mean()) < 1e-8


def test_zero_region_and_support(orthogonal_design):
    X, y, _ = orthogonal_design
    stats = estimate_column_stats(X, fit_intercept=True)
    n = X.shape[0]
    lam_max = np.abs(X.T @ (y - y.mean()) / n).max()
    res = solve_lasso(X, y, stats, lam_max * 1.001, tol=1e-10, max_iter=1000)
    assert res.converged
    assert np.all(res.coef == 0.0)
    # Below lambda_max the largest-|correlation| coordinate enters first.
    res2 = solve_lasso(X, y, stats, lam_max * 0.9, tol=1e-10, max_iter=1000)
    assert np.flatnonzero(res2.coef).size >= 1
