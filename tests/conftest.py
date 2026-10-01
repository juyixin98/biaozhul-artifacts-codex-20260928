"""Shared synthetic fixtures. All data is generated locally with fixed seeds;
no external data or services are involved."""

from __future__ import annotations

import numpy as np
import pytest

from lasso_path.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def orthogonal_design():
    """Columns exactly centered and orthonormal with ||X_j||^2 / n = 1.

    With this design the lasso has the closed form
    beta_j = S(X_j^T (y - y_mean) / n, lambda), which the tests evaluate
    independently of the solver.
    """
    rng = np.random.default_rng(7)
    n, p = 64, 6
    M = rng.standard_normal((n, p))
    M -= M.mean(axis=0)
    Q, _ = np.linalg.qr(M)  # columns stay in the centered subspace
    X = Q * np.sqrt(n)
    beta_true = np.array([3.0, -2.0, 0.0, 1.5, 0.0, 0.0])
    y = X @ beta_true + 0.01 * rng.standard_normal(n)
    return X, y, beta_true


@pytest.fixture
def sparse_ground_truth():
    """Known sparse coefficient vector, well-separated from noise."""
    rng = np.random.default_rng(20260928)
    n, p = 600, 60
    support = np.array([1, 5, 11, 23, 37, 42])
    values = np.array([2.5, -1.8, 3.2, 1.2, -2.7, 0.9])
    beta_true = np.zeros(p)
    beta_true[support] = values
    X = rng.standard_normal((n, p))
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    return X, y, beta_true, support


@pytest.fixture
def correlated_design():
    """Groups of strongly correlated columns (rho ~ 0.98)."""
    rng = np.random.default_rng(1234)
    n, n_groups, group_size = 300, 4, 3
    blocks = []
    for _ in range(n_groups):
        base = rng.standard_normal(n)
        cols = [base + 0.14 * rng.standard_normal(n) for _ in range(group_size)]
        blocks.append(np.column_stack(cols))
    X = np.hstack(blocks)
    p = X.shape[1]
    beta_true = np.zeros(p)
    beta_true[0] = 2.0
    beta_true[6] = -1.5
    y = X @ beta_true + 0.05 * rng.standard_normal(n)
    return X, y, beta_true


@pytest.fixture
def constant_column_design():
    """Design with two constant columns at indices 3 and 7."""
    rng = np.random.default_rng(99)
    n, p = 120, 10
    X = rng.standard_normal((n, p))
    X[:, 3] = 5.0
    X[:, 7] = -2.5
    y = X[:, 0] * 1.5 - X[:, 2] * 0.5 + 0.05 * rng.standard_normal(n)
    return X, y, [3, 7]
