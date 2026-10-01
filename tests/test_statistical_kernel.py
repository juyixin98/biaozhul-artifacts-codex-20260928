from __future__ import annotations

import numpy as np

from app.fixtures import build_fixtures
from app.sample_contract import prepare_sample
from app.statistical_kernel import diagnose_matrix, robust_tyler_shrinkage


def test_independent_symmetry_and_psd_checks_do_not_repair_inputs() -> None:
    asymmetric = np.array([[1.0, 0.2], [0.8, 1.0]])
    negative_symmetric = np.diag([1.0, -2.0])

    asymmetric_report = diagnose_matrix(asymmetric)
    negative_report = diagnose_matrix(negative_symmetric)

    assert asymmetric_report.symmetric is False
    assert any(item.startswith("SYMMETRY:") for item in asymmetric_report.validation_failures)
    assert asymmetric_report.positive_semidefinite is True
    assert negative_report.symmetric is True
    assert negative_report.positive_semidefinite is False
    assert negative_report.min_eigenvalue == -2.0
    assert negative_report.rank == 1
    assert negative_report.invertible is False
    # The checks must leave the defective matrices untouched.
    assert asymmetric[0, 1] == 0.2
    assert asymmetric[1, 0] == 0.8
    assert negative_symmetric[1, 1] == -2.0


def test_high_dim_tyler_reports_invertible_rank_and_reproducible_values() -> None:
    fixture = build_fixtures()["high_dim_small_sample"]
    prepared = prepare_sample(
        fixture.data.tolist(), assume_centered=False, center_method="median"
    )
    first = robust_tyler_shrinkage(
        prepared.standardized, shrinkage=0.25, max_iterations=1000, tolerance=1e-12
    )
    second = robust_tyler_shrinkage(
        prepared.standardized, shrinkage=0.25, max_iterations=1000, tolerance=1e-12
    )

    assert prepared.standardized.shape == (14, 22)
    assert first.converged is True
    assert first.diagnostics is not None
    assert first.diagnostics.rank == 22
    assert first.diagnostics.invertible is True
    assert first.diagnostics.condition_number > 1.0
    np.testing.assert_allclose(first.covariance, second.covariance, rtol=0, atol=1e-14)


def test_tyler_matches_statsmodels_standardized_fixed_point() -> None:
    from statsmodels.robust.covariance import cov_tyler_regularized

    fixture = build_fixtures()["strong_outliers"]
    prepared = prepare_sample(
        fixture.data.tolist(), assume_centered=False, center_method="median"
    )
    result = robust_tyler_shrinkage(
        prepared.standardized, shrinkage=0.25, max_iterations=1000, tolerance=1e-12
    )
    reference = cov_tyler_regularized(
        prepared.standardized,
        start_cov=np.eye(prepared.standardized.shape[1]),
        shrinkage_factor=0.25,
        maxiter=1000,
        eps=1e-13,
    )

    np.testing.assert_allclose(result.covariance, reference.cov, atol=2e-10)
    np.testing.assert_allclose(np.trace(result.covariance), 5.0, atol=1e-10)
    # The planted row receives a weight two orders below ordinary rows.
    assert result.weights[0] < 0.01
    assert np.median(result.weights[1:]) > 0.5


def test_zero_shrinkage_preserves_collinear_rank_deficiency() -> None:
    fixture = build_fixtures()["collinear_columns"]
    prepared = prepare_sample(
        fixture.data.tolist(), assume_centered=False, center_method="median"
    )
    result = robust_tyler_shrinkage(
        prepared.standardized, shrinkage=0.0, max_iterations=1000, tolerance=1e-10
    )

    assert result.diagnostics is not None
    assert result.diagnostics.invertible is False
    assert result.diagnostics.rank == 4


def test_non_convergence_is_reported_as_failure_category() -> None:
    rng = np.random.default_rng(11)
    z = rng.normal(size=(8, 6))
    result = robust_tyler_shrinkage(
        z, shrinkage=0.25, max_iterations=1, tolerance=1e-15
    )

    assert result.converged is False
    assert any(item.startswith("NON_CONVERGED:") for item in result.failures)
