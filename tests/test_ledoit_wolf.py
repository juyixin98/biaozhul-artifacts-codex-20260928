from __future__ import annotations

import numpy as np

from app.fixtures import build_fixtures
from app.sample_contract import prepare_sample
from app.statistical_kernel import ledoit_wolf_shrinkage


def independent_lw(z: np.ndarray) -> tuple[float, np.ndarray]:
    n, p = z.shape
    sample = z.T @ z / n
    target_scale = np.trace(sample) / p
    target = target_scale * np.eye(p)
    squared = z * z
    beta = np.sum((squared.T @ squared) / n - sample * sample) / n
    delta = np.sum((sample - target) ** 2) / p
    shrinkage = min(1.0, max(0.0, beta / delta)) if delta else 0.0
    return float(shrinkage), (1.0 - shrinkage) * sample + shrinkage * target


def test_ledoit_wolf_matches_independent_formula_and_inverts_collinear_data() -> None:
    fixture = build_fixtures()["collinear_columns"]
    prepared = prepare_sample(
        fixture.data.tolist(), assume_centered=False, center_method="mean"
    )
    result = ledoit_wolf_shrinkage(prepared.standardized)
    shrinkage, covariance = independent_lw(prepared.standardized)

    assert result.shrinkage == shrinkage
    np.testing.assert_allclose(result.covariance, covariance, rtol=1e-12, atol=1e-12)
    assert result.diagnostics is not None
    assert result.diagnostics.rank == 6
    assert result.diagnostics.invertible is True
