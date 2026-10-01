"""Reproducible local synthetic validation for the Lasso path backend."""

from __future__ import annotations

import json

import numpy as np

from lasso_backend.preprocessing import prepare_data
from lasso_backend.reference import sklearn_lasso_coefficients
from lasso_backend.solver import lasso_path


def build_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260930)
    n_samples = 140
    base = rng.normal(size=(n_samples, 4))
    base -= base.mean(axis=0, keepdims=True)

    x = np.empty((n_samples, 5))
    x[:, 0] = base[:, 0]
    x[:, 1] = base[:, 0] + 0.03 * base[:, 1]
    x[:, 2] = base[:, 2]
    x[:, 3] = base[:, 3]
    noise = rng.normal(size=n_samples)
    x[:, 4] = noise - noise.mean()
    x *= np.array([1.0, 2.0, 0.4, 3.0, 0.8])

    beta = np.array([2.0, 0.0, -1.5, 0.0, 0.8])
    y = x @ beta + 0.42
    lambdas = np.array([0.8, 0.35, 0.15, 0.06])
    return x, y, lambdas


def main() -> None:
    x, y, lambdas = build_problem()
    prepared = prepare_data(x, y)
    result = lasso_path(
        prepared,
        lambdas=lambdas,
        lambda_order="descending",
        warm_start=True,
        max_iter=50_000,
        tol=1e-9,
    )
    reference = sklearn_lasso_coefficients(
        prepared, result.lambdas, max_iter=100_000, tol=1e-12
    )
    max_reference_error = float(np.max(np.abs(result.coefficients - reference)))

    report = {
        "shape": list(x.shape),
        "lambdas": result.lambdas.tolist(),
        "accepted": result.converged,
        "max_reference_error": max_reference_error,
        "points": [
            {
                "lambda": item.lambda_value,
                "converged": item.converged,
                "status": item.status,
                "iterations": item.iterations,
                "kkt_violation": item.kkt_violation,
                "duality_gap": item.duality_gap,
                "support_size": item.support_size,
            }
            for item in result.diagnostics
        ],
        "small_lambda_support": np.flatnonzero(result.support[-1]).tolist(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    assert result.converged is True
    assert max_reference_error < 2e-7
    assert report["small_lambda_support"] == [0, 2, 4]


if __name__ == "__main__":
    main()
