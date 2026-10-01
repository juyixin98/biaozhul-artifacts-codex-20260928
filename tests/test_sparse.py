from __future__ import annotations

import numpy as np
from scipy import sparse

from lasso_backend.preprocessing import prepare_data
from lasso_backend.solver import lasso_path


class ExplodingCSR(sparse.csr_matrix):
    def toarray(self, order=None, out=None):
        raise AssertionError("sparse core must not densify X")


def test_sparse_coordinate_descent_does_not_densify(known_problem):
    x = ExplodingCSR(known_problem["x"])
    prepared = prepare_data(x, known_problem["y"])

    result = lasso_path(
        prepared,
        lambdas=known_problem["lambdas"],
        max_iter=20_000,
        tol=1e-8,
    )

    dense_prepared = prepare_data(known_problem["x"], known_problem["y"])
    dense_result = lasso_path(
        dense_prepared,
        lambdas=known_problem["lambdas"],
        max_iter=20_000,
        tol=1e-8,
    )

    np.testing.assert_allclose(
        result.coefficients, dense_result.coefficients, atol=1e-9
    )
    np.testing.assert_allclose(
        result.intercepts, dense_result.intercepts, atol=1e-9
    )
    assert sparse.issparse(prepared.x)


def test_sparse_zero_variance_detection_uses_column_aggregates():
    x = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    )
    y = np.array([1.0, 2.0, 1.5, 3.0])

    prepared = prepare_data(
        x, y, zero_variance_policy="zero_fill"
    )
    assert prepared.zero_variance.tolist() == [True, False]

    result = lasso_path(prepared, lambdas=[0.1])
    assert result.coefficients[0, 0] == 0.0
    assert result.diagnostics[0].converged is True
