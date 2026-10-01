"""Coordinate-descent solver for a single lasso problem.

Objective (original scale, intercept never penalized)::

    minimize  (1 / (2 n)) ||y - X w - b||_2^2 + lambda * ||scale * w||_1

which is the standardized problem ``(1/2n)||y_c - X_std beta~||^2 +
lambda ||beta~||_1`` with ``beta~_j = scale_j * w_j`` and
``X_std_j = (X_j - mean_j) / scale_j``.

The iterate is kept in original scale ``w``; centering is handled implicitly
through the residual identity ``r_c = (y - X w) - b`` with
``b = y_mean - mean^T w``, so a sparse CSC design is never densified.

Coordinate update (soft threshold)::

    rho_j = w_j + g_j / scale_j^2,   g_j = (1/n) X_j^T R - mean_j * mean(R)
    w_j  <- S(rho_j, lambda / scale_j)

Convergence is declared from the duality gap; the KKT residual is computed
alongside and reported either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .preprocessing import ColumnStats


def soft_threshold(z: float, a: float) -> float:
    """S(z, a) = sign(z) * max(|z| - a, 0)."""
    if z > a:
        return z - a
    if z < -a:
        return z + a
    return 0.0


@dataclass
class SolverResult:
    coef: np.ndarray  # (p,) original scale
    intercept: float
    lambda_: float
    n_iter: int
    converged: bool
    duality_gap: float
    kkt_violation: float


def _col_dot(X, j: int, vec: np.ndarray) -> float:
    if sparse.issparse(X):
        s, e = X.indptr[j], X.indptr[j + 1]
        return float(np.dot(X.data[s:e], vec[X.indices[s:e]]))
    return float(X[:, j] @ vec)


def _col_axpy(X, j: int, alpha: float, vec: np.ndarray) -> None:
    """vec += alpha * X[:, j], sparse-safe."""
    if sparse.issparse(X):
        s, e = X.indptr[j], X.indptr[j + 1]
        vec[X.indices[s:e]] += alpha * X.data[s:e]
    else:
        vec += alpha * X[:, j]


def _matvec_T(X, vec: np.ndarray) -> np.ndarray:
    return np.asarray(X.T @ vec).ravel()


def _gap_and_kkt(X, Rw, b, w, stats: ColumnStats, lambda_, y_c, n):
    """Duality gap and KKT residual at the current iterate."""
    mean_Rw = Rw.sum() / n
    g = _matvec_T(X, Rw) / n - stats.mean * mean_Rw  # (1/n) X_std^T r_c * scale
    g_hat = g / stats.scale  # gradient in standardized scale
    beta_tilde = stats.scale * w
    r_c = Rw - b

    primal = 0.5 / n * float(r_c @ r_c) + lambda_ * float(np.abs(beta_tilde).sum())
    nu = float(np.abs(g_hat).max(initial=0.0)) / lambda_
    theta = r_c / (n * lambda_ * max(nu, 1.0))
    shifted = theta - y_c / (n * lambda_)
    dual = 0.5 / n * float(y_c @ y_c) - 0.5 * n * lambda_**2 * float(shifted @ shifted)
    gap = primal - dual

    active = beta_tilde != 0.0
    kkt = 0.0
    if active.any():
        kkt = float(np.abs(g_hat[active] - lambda_ * np.sign(beta_tilde[active])).max())
    if (~active).any():
        kkt = max(kkt, float(np.maximum(np.abs(g_hat[~active]) - lambda_, 0.0).max()))
    return gap, kkt


def solve_lasso(
    X,
    y: np.ndarray,
    stats: ColumnStats,
    lambda_: float,
    *,
    tol: float,
    max_iter: int,
    warm_start: np.ndarray | None = None,
) -> SolverResult:
    """Solve one lasso problem. ``stats`` are frozen preprocessing parameters.

    Sparse CSC/CSR input is used without densification; dense input is kept
    in Fortran order for cache-friendly column access.
    """
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    n, p = X.shape
    if sparse.issparse(X):
        Xw = X if sparse.isspmatrix_csc(X) else X.tocsc()
    else:
        Xw = np.asfortranarray(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)

    x_mean = stats.mean
    x_scale = stats.scale
    col_sum = n * x_mean
    y_mean = float(y.mean()) if stats.fit_intercept else 0.0
    y_c = y - y_mean

    w = np.zeros(p) if warm_start is None else np.array(warm_start, dtype=np.float64, copy=True)
    Rw = y - np.asarray(Xw @ w).ravel()  # residual of the *uncentered* system
    b = y_mean - float(x_mean @ w)  # intercept in original scale
    sum_Rw = float(Rw.sum())

    gap_scale = max(1.0, float(y_c @ y_c) / (2.0 * n))
    threshold = tol * gap_scale

    converged = False
    gap, kkt = np.inf, np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        for j in range(p):
            g_j = _col_dot(Xw, j, Rw) / n - x_mean[j] * (sum_Rw / n)
            rho = w[j] + g_j / (x_scale[j] ** 2)
            w_new = soft_threshold(rho, lambda_ / x_scale[j])
            delta = w_new - w[j]
            if delta != 0.0:
                _col_axpy(Xw, j, -delta, Rw)
                b -= delta * x_mean[j]
                sum_Rw -= delta * col_sum[j]
                w[j] = w_new
        gap, kkt = _gap_and_kkt(Xw, Rw, b, w, stats, lambda_, y_c, n)
        if gap <= threshold:
            converged = True
            break

    return SolverResult(
        coef=w,
        intercept=float(b),
        lambda_=lambda_,
        n_iter=n_iter,
        converged=converged,
        duality_gap=float(gap),
        kkt_violation=float(kkt),
    )
