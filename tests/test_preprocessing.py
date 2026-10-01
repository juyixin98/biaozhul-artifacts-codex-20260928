from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from lasso_backend.preprocessing import PreprocessingError, prepare_data


def test_frozen_centering_scaling_and_intercept_mapping(known_problem):
    prepared = prepare_data(known_problem["x"], known_problem["y"])

    assert prepared.x.shape == (140, 5)
    assert np.max(np.abs(prepared.x.mean(axis=0))) == pytest.approx(0.0, abs=1e-14)
    assert np.max(np.abs(prepared.x.std(axis=0, ddof=1) - 1.0)) < 1e-14
    assert prepared.y_mean == pytest.approx(0.42)
    np.testing.assert_allclose(
        prepared.y_centered, known_problem["y"] - 0.42, atol=1e-12
    )

    beta = np.array([1.0, -2.0, 3.0, -4.0, 0.5])
    assert prepared.to_original_coefficients(beta * prepared.scales) == pytest.approx(
        beta
    )
    assert prepared.original_intercept(beta * prepared.scales) == pytest.approx(
        0.42 - np.dot(prepared.x_means, beta)
    )


def test_constant_column_is_rejected_with_index_category():
    x = np.arange(12, dtype=float).reshape(6, 2)
    x[:, 1] = 7.0
    y = np.arange(6, dtype=float)

    with pytest.raises(PreprocessingError, match="columns rejected: \\[1\\]"):
        prepare_data(x, y)


def test_zero_fill_policy_keeps_inactive_zero_variance_column():
    x = np.arange(12, dtype=float).reshape(6, 2)
    x[:, 1] = -1.0
    y = np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0])

    prepared = prepare_data(
        x, y, zero_variance_policy="zero_fill"
    )
    assert prepared.zero_variance.tolist() == [False, True]
    assert np.count_nonzero(prepared.x[:, 1]) == 0
    assert prepared.coordinate_denominator(1) == 0.0


def test_invalid_dimensions_and_nonfinite_values_are_rejected():
    y = np.arange(5, dtype=float)
    with pytest.raises(PreprocessingError, match="same row count"):
        prepare_data(np.zeros((4, 2)), y)

    bad = sparse.csr_matrix(np.eye(3))
    bad.data[0] = np.nan
    with pytest.raises(PreprocessingError, match="NaN or infinite"):
        prepare_data(bad, y[:3])
