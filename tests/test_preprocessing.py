"""Frozen preprocessing parameters and scale restoration."""

import numpy as np

from lasso_path.preprocessing import (estimate_column_stats, lambda_max,
                                      restore_coef, restore_intercept,
                                      standardize_coef)
from lasso_path.solver import solve_lasso


def test_stats_match_independent_numpy(sparse_ground_truth):
    X, y, _, _ = sparse_ground_truth
    stats = estimate_column_stats(X, fit_intercept=True)
    # Independent recomputation in the test (not via the package).
    assert np.allclose(stats.mean, X.mean(axis=0))
    assert np.allclose(stats.scale, X.std(axis=0))  # population std, ddof=0
    assert stats.fit_intercept is True


def test_stats_frozen_and_not_mutated(sparse_ground_truth, settings):
    X, y, _, _ = sparse_ground_truth
    stats = estimate_column_stats(X, fit_intercept=True)
    mean_before, scale_before = stats.mean.copy(), stats.scale.copy()
    solve_lasso(X, y, stats, 0.05, tol=1e-8, max_iter=settings.default_max_iter)
    # Solving must not re-estimate or mutate the frozen parameters.
    assert np.array_equal(stats.mean, mean_before)
    assert np.array_equal(stats.scale, scale_before)


def test_scale_restoration_roundtrip(sparse_ground_truth):
    X, _, _, _ = sparse_ground_truth
    stats = estimate_column_stats(X, fit_intercept=True)
    rng = np.random.default_rng(3)
    w = rng.standard_normal(X.shape[1])
    beta_tilde = standardize_coef(w, stats)
    assert np.allclose(restore_coef(beta_tilde, stats), w)
    assert np.allclose(beta_tilde, stats.scale * w)


def test_restored_prediction_matches_standardized(sparse_ground_truth, settings):
    """Predictions from restored (original-scale) coefficients must equal
    predictions computed on the standardized design."""
    X, y, _, _ = sparse_ground_truth
    stats = estimate_column_stats(X, fit_intercept=True)
    res = solve_lasso(X, y, stats, 0.05, tol=1e-9, max_iter=5000)
    beta_tilde = standardize_coef(res.coef, stats)
    X_std = (X - stats.mean) / stats.scale
    pred_original = X @ res.coef + res.intercept
    pred_standardized = X_std @ beta_tilde + y.mean()
    assert np.allclose(pred_original, pred_standardized, atol=1e-8)
    # Intercept identity: b = y_mean - mean^T w.
    assert abs(res.intercept - (y.mean() - stats.mean @ res.coef)) < 1e-10


def test_restore_intercept_matches_solver(sparse_ground_truth, settings):
    X, y, _, _ = sparse_ground_truth
    stats = estimate_column_stats(X, fit_intercept=True)
    res = solve_lasso(X, y, stats, 0.05, tol=1e-9, max_iter=5000)
    beta_tilde = standardize_coef(res.coef, stats)
    b = restore_intercept(beta_tilde, y.mean(), stats)
    assert abs(b - res.intercept) < 1e-10


def test_lambda_max_independent(sparse_ground_truth):
    X, y, _, _ = sparse_ground_truth
    stats = estimate_column_stats(X, fit_intercept=True)
    n = X.shape[0]
    y_c = y - y.mean()
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    expected = np.abs(X_std.T @ y_c).max() / n
    assert abs(lambda_max(X, y, stats) - expected) < 1e-12
