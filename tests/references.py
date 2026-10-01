"""Independent references used only by tests."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def linear_program_ot(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Small exact LP reference, independent of the Sinkhorn core."""

    n, m = cost.shape
    equality = np.vstack(
        [
            np.repeat(np.eye(n), m, axis=1),
            np.tile(np.eye(m), n),
        ]
    )
    result = linprog(
        cost.reshape(-1),
        A_eq=equality,
        b_eq=np.concatenate([a, b]),
        bounds=[(0.0, None)] * (n * m),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP reference failed: {result.message}")
    plan = result.x.reshape(n, m)
    return plan, float(result.fun)


def multiplicative_sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
    *,
    max_iterations: int = 100_000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int]:
    """Classical probability-domain scaling reference.

    It intentionally uses direct kernel exponentiation and u/v scaling instead
    of the implementation under test's log-domain dual updates.
    """

    kernel = np.exp(-cost / epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)
    active_a = a > 0.0
    active_b = b > 0.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        denominator = kernel @ v
        u[active_a] = a[active_a] / denominator[active_a]
        denominator = kernel.T @ u
        v[active_b] = b[active_b] / denominator[active_b]
        plan = (u[:, None] * kernel * v[None, :])
        row_error = float(np.max(np.abs(plan.sum(axis=1) - a)))
        col_error = float(np.max(np.abs(plan.sum(axis=0) - b)))
        if row_error <= tolerance and col_error <= tolerance:
            return plan, iterations
    raise AssertionError("multiplicative reference did not converge")

