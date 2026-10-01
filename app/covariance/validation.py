"""Independent matrix validation and numerical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import settings
from app.errors import MatrixValidationError


@dataclass(frozen=True)
class MatrixDiagnostics:
    symmetric: bool
    positive_semidefinite: bool
    rank: int
    invertible: bool
    n_features: int
    min_eigenvalue: float
    max_eigenvalue: float
    eigenvalues: list[float]
    condition_number: float | None
    symmetry_max_abs_error: float
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symmetric": self.symmetric,
            "positive_semidefinite": self.positive_semidefinite,
            "rank": self.rank,
            "invertible": self.invertible,
            "n_features": self.n_features,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "eigenvalues": self.eigenvalues,
            "condition_number": self.condition_number,
            "symmetry_max_abs_error": self.symmetry_max_abs_error,
            "checks": self.checks,
            "warnings": self.warnings,
        }


def validate_covariance(
    matrix: np.ndarray,
    *,
    method: str,
    symmetry_tolerance: float = settings.symmetry_tolerance,
    eigenvalue_tolerance: float = settings.eigenvalue_tolerance,
) -> MatrixDiagnostics:
    """Validate symmetry and PSD separately; never repair eigenvalues."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise MatrixValidationError(
            "Covariance must be a square matrix.",
            stage="matrix_shape",
            details={"shape": list(array.shape), "method": method},
        )

    symmetry_error = float(np.max(np.abs(array - array.T)))
    symmetric = bool(np.isfinite(symmetry_error) and symmetry_error <= symmetry_tolerance)
    if not symmetric:
        raise MatrixValidationError(
            "Symmetry validation failed independently of PSD validation.",
            stage="symmetry_validation",
            details={
                "method": method,
                "symmetry_max_abs_error": symmetry_error,
                "tolerance": symmetry_tolerance,
            },
        )

    # eigh assumes symmetry, so it is called only after the symmetry check.
    eigenvalues = np.linalg.eigvalsh(0.5 * (array + array.T))
    finite = bool(np.all(np.isfinite(eigenvalues)))
    min_eigenvalue = float(eigenvalues.min()) if finite else float("nan")
    max_eigenvalue = float(eigenvalues.max()) if finite else float("nan")
    psd = bool(finite and min_eigenvalue >= -eigenvalue_tolerance)
    if not psd:
        raise MatrixValidationError(
            "Positive semidefinite validation failed; eigenvalues were not clipped.",
            stage="psd_validation",
            details={
                "method": method,
                "min_eigenvalue": min_eigenvalue,
                "tolerance": eigenvalue_tolerance,
                "finite": finite,
            },
        )

    # SVD rank is a separate rank calculation from the eigenvalue sign test.
    rank = int(np.linalg.matrix_rank(array, hermitian=True))
    positive_eigenvalues = eigenvalues[eigenvalues > eigenvalue_tolerance]
    invertible = bool(rank == array.shape[0] and positive_eigenvalues.size == array.shape[0])
    if invertible:
        condition_number = float(positive_eigenvalues.max() / positive_eigenvalues.min())
    else:
        condition_number = None

    warnings: list[str] = []
    if not invertible:
        warnings.append("Matrix is singular or numerically rank deficient; no ordinary inverse exists.")
    if abs(min_eigenvalue) <= eigenvalue_tolerance:
        warnings.append("A zero or tolerance-sized eigenvalue is present.")

    return MatrixDiagnostics(
        symmetric=True,
        positive_semidefinite=True,
        rank=rank,
        invertible=invertible,
        n_features=int(array.shape[0]),
        min_eigenvalue=float(eigenvalues.min()),
        max_eigenvalue=float(eigenvalues.max()),
        eigenvalues=[float(value) for value in eigenvalues],
        condition_number=condition_number,
        symmetry_max_abs_error=symmetry_error,
        checks=[
            "shape is square",
            f"max |A-A^T| <= {symmetry_tolerance}",
            "eigvalsh used only after symmetry check",
            f"all eigenvalues >= {-eigenvalue_tolerance}",
            "rank computed separately with SVD",
        ],
        warnings=warnings,
    )
