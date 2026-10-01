from __future__ import annotations

import numpy as np
import pytest

from app.covariance.validation import validate_covariance
from app.errors import MatrixValidationError


def test_symmetry_and_psd_failures_are_separate_stages() -> None:
    asymmetric = np.array([[1.0, 0.2], [0.0, 1.0]])
    with pytest.raises(MatrixValidationError) as symmetry_error:
        validate_covariance(asymmetric, method="test")
    assert symmetry_error.value.stage == "symmetry_validation"
    assert symmetry_error.value.details["symmetry_max_abs_error"] == pytest.approx(0.2)

    indefinite = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(MatrixValidationError) as psd_error:
        validate_covariance(indefinite, method="test")
    assert psd_error.value.stage == "psd_validation"
    assert psd_error.value.details["min_eigenvalue"] == pytest.approx(-1.0)


def test_singular_matrix_reports_rank_invertibility_and_condition_as_none() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    diagnostics = validate_covariance(singular, method="test")
    assert diagnostics.symmetric is True
    assert diagnostics.positive_semidefinite is True
    assert diagnostics.rank == 1
    assert diagnostics.invertible is False
    assert diagnostics.condition_number is None
    assert diagnostics.min_eigenvalue == pytest.approx(0.0, abs=1e-12)


def test_positive_definite_matrix_reports_condition_number() -> None:
    matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    diagnostics = validate_covariance(matrix, method="test")
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert diagnostics.rank == 2
    assert diagnostics.invertible is True
    assert diagnostics.condition_number == pytest.approx(eigenvalues.max() / eigenvalues.min())
