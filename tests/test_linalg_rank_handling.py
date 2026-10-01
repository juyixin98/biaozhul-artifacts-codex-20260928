import numpy as np

from app.config import Tolerances
from app.core.linalg import is_row_independent, rank_and_basis, solve_face_kkt


def test_rank_and_basis_detects_exact_duplicates_without_inverse():
    A = np.asarray([[1.0, 1.0], [2.0, 2.0], [1.0, 0.0]])

    rank, independent, basis, dependent = rank_and_basis(A, rank_tol=1e-12)

    assert rank == 2
    assert set(independent.tolist()) == {0, 2} or set(independent.tolist()) == {1, 2}
    assert dependent.tolist() in ([1], [0])
    assert not is_row_independent(A[[0, 2]], np.asarray([-3.0, -3.0]), rank_tol=1e-12)


def test_rank_deficient_face_returns_zero_direction_and_min_norm_multipliers():
    H = np.eye(2) * 2.0
    gradient = np.asarray([-2.0, -4.0])
    face = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])

    d, multipliers, stationarity, info = solve_face_kkt(
        H, gradient, face, rank_tol=1e-12, step_tol=1e-10
    )

    np.testing.assert_allclose(d, [0.0, 0.0], atol=1e-12)
    assert stationarity < 1e-10
    assert info["face_rank"] == 2
    assert info["face_rows"] == 3
    assert info["zero_direction"] is True
