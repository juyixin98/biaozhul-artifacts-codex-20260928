import numpy as np
from scipy.linalg import qr

from app.errors import NumericalError


def rank_and_basis(A: np.ndarray, rank_tol: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return rank, independent row indices, row orthonormal basis, and dependent indices.

    Dense SVD detects rank explicitly; equality rows are never inverted blindly.
    A rank-revealing QR row permutation selects representative independent rows.
    """
    if A.size == 0:
        return 0, np.empty(0, dtype=int), np.empty((0, A.shape[1] if A.ndim == 2 else 0)), np.empty(0, dtype=int)
    u, singular_values, _ = np.linalg.svd(A, full_matrices=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    threshold = rank_tol * max(1.0, scale)
    r = int(np.sum(singular_values > threshold))
    if r == 0:
        return 0, np.empty(0, dtype=int), np.empty((0, A.shape[1])), np.arange(A.shape[0])

    q, r_matrix, pivots = qr(A.T, mode="economic", pivoting=True)  # noqa: N806
    row_scale = np.max(np.abs(np.diag(r_matrix))) if min(r_matrix.shape) else 0.0
    qr_threshold = rank_tol * max(1.0, row_scale)
    diag_abs = np.abs(np.diag(r_matrix, k=0))
    independent = np.asarray(pivots[: int(np.sum(diag_abs > qr_threshold))], dtype=int)
    if independent.size < r:
        independent = np.asarray(pivots[:r], dtype=int)
    independent = independent[:r]
    dependent = np.asarray([i for i in range(A.shape[0]) if i not in set(independent.tolist())], dtype=int)
    basis = u[:, :r].T
    return r, independent, basis, dependent


def null_space(A: np.ndarray, rank_tol: float) -> tuple[np.ndarray, int, np.ndarray]:
    """Return a null-space matrix Z, rank r, and singular values of A."""
    n = A.shape[1]
    if A.shape[0] == 0:
        return np.eye(n), 0, np.asarray([])
    _, s, vt = np.linalg.svd(A, full_matrices=True)
    scale = float(s[0]) if s.size else 0.0
    threshold = rank_tol * max(1.0, scale)
    r = int(np.sum(s > threshold))
    if r == n:
        return np.empty((n, 0)), r, s
    return vt[r:, :].T, r, s


def solve_face_kkt(
    H: np.ndarray,
    gradient: np.ndarray,
    A_face: np.ndarray,
    rank_tol: float,
    step_tol: float,
    x_scale: float = 1.0,
):
    """Minimize 0.5 d'Hd + g'd subject to A_face d=0.

    The solution uses the null-space method:
      1. SVD gives the constraint null space and rank.
      2. Cholesky solves the positive-definite reduced system.
      3. A minimum-norm range-space correction gives multipliers.
    """
    n = H.shape[0]
    Z, rank, singular_values = null_space(A_face, rank_tol)
    info = {
        "face_rank": int(rank),
        "face_rows": int(A_face.shape[0]),
        "singular_values": singular_values.tolist(),
    }

    if Z.shape[1] == 0:
        d = np.zeros(n)
        # Full row rank in column space; solve A_face^T lambda = -g.
        y, *_ = np.linalg.lstsq(A_face.T, -gradient, rcond=None)
        multipliers = np.asarray(y, dtype=float)
        stationarity = H @ d + gradient + A_face.T @ multipliers
        info["multiplier_solve"] = "lstsq_full_span"
        info["zero_direction"] = True
        return d, multipliers, float(np.linalg.norm(stationarity, ord=np.inf)), info

    Hz = H @ Z
    reduced_H = Z.T @ Hz
    reduced_g = Z.T @ gradient
    try:
        L = np.linalg.cholesky(0.5 * (reduced_H + reduced_H.T))
        y = np.linalg.solve(L, -reduced_g)
        z = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError as exc:
        raise NumericalError("reduced_hessian_not_positive_definite", str(exc)) from exc

    d_null = Z @ z
    d = d_null
    if A_face.shape[0]:
        rhs = -(H @ d + gradient)
        multipliers, *_ = np.linalg.lstsq(A_face.T, rhs, rcond=None)
        multipliers = np.asarray(multipliers, dtype=float)
        stationarity = H @ d + gradient + A_face.T @ multipliers
        info["multiplier_solve"] = "lstsq_rank_deficient_minimum_norm"
    else:
        multipliers = np.asarray([])
        stationarity = H @ d + gradient
        info["multiplier_solve"] = "unconstrained"

    d_norm = float(np.linalg.norm(d))
    x_scale = max(1.0, float(x_scale))
    scale = max(1.0, float(np.linalg.norm(gradient)))
    info["step_norm"] = d_norm
    info["zero_direction"] = d_norm <= step_tol * x_scale
    return d, multipliers, float(np.linalg.norm(stationarity, ord=np.inf)), info


def is_row_independent(face: np.ndarray, row: np.ndarray, rank_tol: float) -> bool:
    if face.shape[0] == 0:
        return float(np.linalg.norm(row)) > rank_tol
    old_rank, _, _, _ = rank_and_basis(face, rank_tol)
    stacked = np.vstack([face, row.reshape(1, -1)])
    new_rank, _, _, _ = rank_and_basis(stacked, rank_tol)
    return new_rank > old_rank
