from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fixture:
    name: str
    description: str
    data: np.ndarray


def build_fixtures() -> dict[str, Fixture]:
    """Deterministic local fixtures covering the requested failure modes."""
    rng = np.random.default_rng(20261001)

    n, p = 14, 22
    base = rng.normal(size=(n, p))
    transform = rng.normal(size=(p, p)) * 0.08
    high_dim = base @ transform + rng.normal(scale=0.03, size=(n, p))

    collinear = rng.normal(size=(24, 6))
    collinear[:, 2] = 2.0 * collinear[:, 0] - 3.0 * collinear[:, 1]
    collinear[:, 4] = -0.5 * collinear[:, 3]

    outliers = rng.normal(size=(30, 5))
    outliers[0, :] += np.array([18.0, -15.0, 20.0, 0.0, 12.0])
    outliers[1, 0] -= 22.0
    outliers[2, 2] += 25.0

    rescaled = outliers.copy()
    rescaled *= np.array([1000.0, 0.01, 50.0, 1.0, 300.0])

    missing = rng.normal(size=(18, 8))
    missing[0, 0] = np.nan
    missing[5, 2] = np.nan
    missing[17, 7] = np.nan

    return {
        "high_dim_small_sample": Fixture(
            "high_dim_small_sample",
            "14 observations and 22 dimensions to test rank and regularization.",
            high_dim,
        ),
        "collinear_columns": Fixture(
            "collinear_columns",
            "Exact linear column dependence makes the empirical covariance singular.",
            collinear,
        ),
        "strong_outliers": Fixture(
            "strong_outliers",
            "Three strongly radial or coordinatewise contaminated rows.",
            outliers,
        ),
        "rescaled_outliers": Fixture(
            "rescaled_outliers",
            "The outlier fixture with highly unequal fixed column scales.",
            rescaled,
        ),
        "missing_values": Fixture(
            "missing_values",
            "Three null cells for deterministic median-imputation checks.",
            missing,
        ),
    }
