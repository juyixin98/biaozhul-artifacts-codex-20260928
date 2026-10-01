from __future__ import annotations

import numpy as np
import pytest
from statsmodels.robust.covariance import cov_ogk

from app.covariance.estimators import (
    empirical_covariance,
    ledoit_wolf_covariance,
    ogk_covariance,
)
from app.covariance.sample import prepare_sample
from tests.fixtures import (
    clean_rescaled_dataset,
    collinear_dataset,
    high_dimensional_dataset,
    independent_empirical_covariance,
    independent_ledoit_wolf,
    strong_outlier_dataset,
)


def test_empirical_matches_independent_formula_and_reports_rescaled_variance() -> None:
    data = clean_rescaled_dataset()
    result = empirical_covariance(prepare_sample(data.tolist()))
    location, expected = independent_empirical_covariance(data)

    assert np.allclose(result.location, location)
    assert np.allclose(result.covariance, expected)
    assert result.diagnostics.rank == 4
    assert result.diagnostics.invertible is True
    assert result.covariance[3, 3] > 1000.0 * result.covariance[0, 0]


def test_ledoit_wolf_matches_independent_formula_and_inverts_high_dimensional_sample() -> None:
    data = high_dimensional_dataset()
    sample = prepare_sample(data.tolist())
    empirical = empirical_covariance(sample)
    result = ledoit_wolf_covariance(sample)
    location, expected, expected_shrinkage = independent_ledoit_wolf(data)

    assert np.allclose(result.location, location)
    assert np.allclose(result.covariance, expected, atol=1e-12)
    assert result.parameters["shrinkage"] == np.float64(expected_shrinkage)
    assert empirical.diagnostics.rank == 7
    assert empirical.diagnostics.invertible is False
    assert result.diagnostics.rank == 12
    assert result.diagnostics.invertible is True
    assert result.diagnostics.min_eigenvalue > 0.3
    assert result.diagnostics.condition_number < 4.0


def test_ogk_matches_mature_statsmodels_implementation_on_full_rank_rescaled_data() -> None:
    data = clean_rescaled_dataset()
    ours = ogk_covariance(prepare_sample(data.tolist()))
    reference = cov_ogk(data, maxiter=2, reweight=None, rescale_raw=True)

    assert np.allclose(ours.covariance, reference.cov_raw, atol=1e-9, rtol=1e-9)
    assert np.allclose(ours.location, reference.loc_raw, atol=1e-9, rtol=1e-9)
    assert ours.diagnostics.rank == 4
    assert ours.diagnostics.invertible is True


def test_ogk_reports_strong_outlier_weights_and_resists_scale_pollution() -> None:
    data, outlier_indices = strong_outlier_dataset()
    empirical = empirical_covariance(prepare_sample(data.tolist()))
    ours = ogk_covariance(prepare_sample(data.tolist()))
    clean = ogk_covariance(prepare_sample(clean_rescaled_dataset().tolist()))

    zero_weights = {weight.index for weight in ours.weights if weight.weight == 0.0}
    assert zero_weights == set(outlier_indices)
    assert all(ours.weights[i].robust_distance > ours.weights[i].threshold for i in outlier_indices)
    assert all(weight.weight == 1.0 for i, weight in enumerate(ours.weights) if i not in outlier_indices)
    assert np.allclose(ours.covariance, clean.covariance, atol=0.12, rtol=0.08)
    assert empirical.covariance[3, 3] > 1.4 * ours.covariance[3, 3]
    assert ours.diagnostics.eigenvalues == sorted(ours.diagnostics.eigenvalues)
    assert ours.diagnostics.condition_number < 350_000


def test_collinear_columns_keep_zero_eigenvalue_without_negative_repair() -> None:
    data = collinear_dataset()
    empirical = empirical_covariance(prepare_sample(data.tolist()))
    ours = ogk_covariance(prepare_sample(data.tolist()))

    assert empirical.diagnostics.rank == 3
    assert empirical.diagnostics.invertible is False
    assert empirical.diagnostics.condition_number is None
    assert ours.diagnostics.rank == 3
    assert ours.diagnostics.invertible is False
    assert ours.diagnostics.min_eigenvalue == pytest.approx(0.0, abs=1e-12)
    assert max(abs(eigenvalue) for eigenvalue in ours.diagnostics.eigenvalues[:1]) < 1e-12


def test_every_method_reports_assumptions_and_audit_steps() -> None:
    sample = prepare_sample(clean_rescaled_dataset(20).tolist())
    for result in [
        empirical_covariance(sample),
        ledoit_wolf_covariance(sample),
        ogk_covariance(sample),
    ]:
        assert result.assumptions
        assert len(result.weights) == 20
        assert {step["name"] for step in result.steps}
        assert all("validate" in step["name"] for step in result.steps[-1:])
