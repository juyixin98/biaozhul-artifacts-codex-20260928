from __future__ import annotations

import numpy as np
import pytest

from lasso_backend.preprocessing import prepare_data
from lasso_backend.reference import sklearn_lasso_coefficients
from lasso_backend.solver import SolverError, lasso_path


def test_warm_start_path_matches_sklearn_and_restores_scale(known_problem):
    prepared = prepare_data(known_problem["x"], known_problem["y"])
    lambdas = known_problem["lambdas"]

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

    assert result.lambdas.tolist() == pytest.approx(
        np.sort(lambdas)[::-1].tolist()
    )
    assert result.converged is True
    assert [item.status for item in result.diagnostics] == ["converged"] * 4
    assert all(item.kkt_violation <= 1e-9 for item in result.diagnostics)
    assert all(item.duality_gap <= 1e-8 for item in result.diagnostics)
    np.testing.assert_allclose(result.coefficients, reference, atol=2e-7)

    # At the smallest penalty the known support is recovered; correlated twin
    # column 1 stays excluded while its partner column 0 is selected.
    assert result.support[-1].tolist() == [True, False, True, False, True]
    np.testing.assert_allclose(
        result.coefficients[-1],
        reference[-1],
        atol=2e-7,
    )
    assert result.intercepts[-1] == pytest.approx(
        known_problem["intercept"], abs=0.02
    )

    low_penalty = lasso_path(
        prepared,
        lambdas=[1e-5],
        warm_start=False,
        max_iter=50_000,
        tol=1e-10,
    )
    assert low_penalty.converged is True
    np.testing.assert_allclose(
        low_penalty.coefficients[0], known_problem["beta"], atol=0.02
    )
    assert low_penalty.intercepts[0] == pytest.approx(
        known_problem["intercept"], abs=0.01
    )


def test_lambda_order_changes_schedule_without_chaining_answers(known_problem):
    prepared = prepare_data(known_problem["x"], known_problem["y"])
    lambdas = known_problem["lambdas"]

    descending = lasso_path(
        prepared, lambdas=lambdas, lambda_order="descending", tol=1e-9
    )
    ascending = lasso_path(
        prepared, lambdas=lambdas, lambda_order="ascending", tol=1e-9
    )

    assert descending.lambdas[0] > descending.lambdas[-1]
    assert ascending.lambdas[0] < ascending.lambdas[-1]
    descending_by_lambda = descending.coefficients[np.argsort(descending.lambdas)]
    ascending_by_lambda = ascending.coefficients[np.argsort(ascending.lambdas)]
    np.testing.assert_allclose(
        descending_by_lambda, ascending_by_lambda, atol=1e-8
    )


def test_cold_start_matches_warm_start_at_same_lambda(known_problem):
    prepared = prepare_data(known_problem["x"], known_problem["y"])
    kwargs = {
        "lambdas": known_problem["lambdas"],
        "tol": 1e-10,
        "max_iter": 50_000,
    }
    warm = lasso_path(prepared, warm_start=True, **kwargs)
    cold = lasso_path(prepared, warm_start=False, **kwargs)

    np.testing.assert_allclose(warm.coefficients, cold.coefficients, atol=1e-9)


def test_high_lambda_returns_empty_support_and_intercept_only(known_problem):
    prepared = prepare_data(known_problem["x"], known_problem["y"])
    result = lasso_path(prepared, lambdas=[10.0], tol=1e-10)

    assert result.converged is True
    assert result.support[0].sum() == 0
    np.testing.assert_allclose(result.coefficients[0], 0.0, atol=1e-14)
    assert result.intercepts[0] == pytest.approx(0.42)


def test_max_iter_reports_specific_nonconvergence_failure(known_problem):
    prepared = prepare_data(known_problem["x"], known_problem["y"])
    result = lasso_path(
        prepared,
        lambdas=[0.001],
        max_iter=1,
        tol=1e-12,
    )

    assert result.converged is False
    diagnostic = result.diagnostics[0]
    assert diagnostic.converged is False
    assert diagnostic.status == "max_iterations"
    assert diagnostic.iterations == 1
    assert diagnostic.kkt_violation > 1e-12
    assert "KKT violation" in diagnostic.reason


def test_rejects_nonpositive_lambda_and_bad_tolerance(known_problem):
    prepared = prepare_data(known_problem["x"], known_problem["y"])

    with pytest.raises(SolverError, match="strictly positive"):
        lasso_path(prepared, lambdas=[0.0])
    with pytest.raises(SolverError, match="tol must"):
        lasso_path(prepared, lambdas=[0.1], tol=0.0)
