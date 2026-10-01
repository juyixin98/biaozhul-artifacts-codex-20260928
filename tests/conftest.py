from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def known_problem():
    """Noise-free data with strong pair correlation and unequal scales."""
    rng = np.random.default_rng(20260930)
    n_samples = 140
    base = rng.normal(size=(n_samples, 4))

    # Enforce an exact zero sample mean before constructing correlated data.
    base -= base.mean(axis=0, keepdims=True)
    x = np.empty((n_samples, 5))
    x[:, 0] = base[:, 0]
    x[:, 1] = base[:, 0] + 0.03 * base[:, 1]
    x[:, 2] = base[:, 2]
    x[:, 3] = base[:, 3]
    noise = rng.normal(size=n_samples)
    x[:, 4] = noise - noise.mean()
    x *= np.array([1.0, 2.0, 0.4, 3.0, 0.8])

    true_beta = np.array([2.0, 0.0, -1.5, 0.0, 0.8])
    true_intercept = 0.42
    y = x @ true_beta + true_intercept
    lambdas = np.array([0.8, 0.35, 0.15, 0.06])
    return {
        "x": x,
        "y": y,
        "beta": true_beta,
        "intercept": true_intercept,
        "lambdas": lambdas,
    }
