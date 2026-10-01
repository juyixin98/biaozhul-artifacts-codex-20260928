"""Independent centralized quadratic reference used by diagnostics and tests.

This module intentionally does not use the ADMM kernel.  It solves the
aggregated stationarity system for the unconstrained quadratic objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ModelParseError
from .models import QuadraticModel


@dataclass(frozen=True)
class CentralSolution:
    x: np.ndarray
    objective: float
    status: str
    residual: float
    min_eigenvalue: float


def central_solution(
    model: QuadraticModel, *, positive_definite_tol: float = 1e-10
) -> CentralSolution:
    P = np.zeros((model.dimension, model.dimension), dtype=float)
    q = np.zeros(model.dimension, dtype=float)
    for worker in model.workers:
        P += worker.P
        q += worker.q
    constant = float(sum(worker.r for worker in model.workers))
    eigenvalues = np.linalg.eigvalsh(P)
    min_eigenvalue = float(eigenvalues[0])

    if min_eigenvalue > positive_definite_tol:
        x = np.linalg.solve(P, -q)
        residual = float(np.linalg.norm(P @ x + q))
        status = "unique_optimum"
    else:
        x, _, rank, _ = np.linalg.lstsq(P, -q, rcond=None)
        residual = float(np.linalg.norm(P @ x + q))
        if residual > 1e-8 * max(1.0, np.linalg.norm(q)):
            status = "unbounded_below_or_linear_component_in_nullspace"
        elif rank < model.dimension:
            status = "minimum_norm_optimum"
        else:
            status = "unique_optimum"

    objective = float(
        sum(
            0.5 * x @ worker.P @ x + worker.q @ x + worker.r
            for worker in model.workers
        )
    )
    if not np.isfinite(objective):
        raise ModelParseError("central objective is not finite")
    return CentralSolution(
        x=x,
        objective=objective,
        status=status,
        residual=residual,
        min_eigenvalue=min_eigenvalue,
    )
