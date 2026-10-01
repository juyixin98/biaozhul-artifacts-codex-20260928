"""Numerical ADMM primitives.

The scaled-dual form uses::

    x_i = argmin f_i(x_i) + rho/2 ||x_i-z+u_i||^2
    z   = mean_i x_i
    u_i += x_i-z

Every local update is explicitly tied to one round and rho. The coordinator
performs the consensus and dual steps only after both local updates arrive.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import ComputationError
from .models import Problem, QuadraticModel, SolverState


def solve_local_update(
    model: QuadraticModel,
    *,
    z: np.ndarray,
    u: np.ndarray,
    rho: float,
    worker_id: int,
) -> np.ndarray:
    """Solve one strongly convex quadratic local ADMM subproblem."""

    if not math.isfinite(rho) or rho <= 0.0:
        raise ComputationError("rho must be positive for a local update.")
    system = model.Q + rho * np.eye(model.Q.shape[0], dtype=float)
    rhs = -model.q - rho * u + rho * z
    try:
        x = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ComputationError(
            f"Local KKT system for worker {worker_id} could not be solved.",
            {"worker_id": worker_id},
        ) from exc
    if not np.all(np.isfinite(x)):
        raise ComputationError(
            f"Local update for worker {worker_id} produced non-finite values.",
            {"worker_id": worker_id},
        )
    stationarity = system @ x - rhs
    residual = float(np.linalg.norm(stationarity, ord=np.inf))
    scale = max(float(np.linalg.norm(rhs, ord=np.inf)), float(np.linalg.norm(np.diag(system), ord=np.inf)), 1.0)
    if residual > 1e-8 * scale:
        raise ComputationError(
            f"Local KKT residual for worker {worker_id} is too large.",
            {
                "worker_id": worker_id,
                "stationarity_inf_residual": residual,
                "normalised_residual": residual / scale,
            },
        )
    return x


def validate_local_update(
    state: SolverState,
    worker_id: int,
    x: np.ndarray,
    *,
    round_number: int,
    rho: float,
) -> None:
    """Validate that an external process solved the correct round."""

    if round_number != state.round_number:
        from .errors import StateConflict

        raise StateConflict(
            "Submission belongs to a different round.",
            {"submitted_round": round_number, "current_round": state.round_number},
        )
    if not math.isclose(rho, state.rho, rel_tol=0.0, abs_tol=1e-12):
        from .errors import StateConflict

        raise StateConflict(
            "Submission used a different penalty rho.",
            {"submitted_rho": rho, "current_rho": state.rho},
        )
    if x.shape != (state.problem.dimension,):
        from .errors import InputError

        raise InputError(
            "Local update has the wrong shape.",
            {"shape": list(x.shape), "expected": [state.problem.dimension]},
        )
    if not np.all(np.isfinite(x)):
        from .errors import InputError

        raise InputError("Local update contains NaN or infinity.")

    model = state.problem.models[worker_id - 1]
    stationarity = (
        (model.Q + rho * np.eye(state.problem.dimension)) @ x
        + model.q
        + rho * state.u[worker_id - 1]
        - rho * state.z
    )
    residual = float(np.linalg.norm(stationarity, ord=np.inf))
    scale = max(
        float(np.linalg.norm(model.q, ord=np.inf)),
        rho * max(float(np.linalg.norm(state.u[worker_id - 1], ord=np.inf)), 1.0),
        rho,
        rho * max(float(np.linalg.norm(state.z, ord=np.inf)), 1.0),
        float(np.linalg.norm(np.diag(model.Q), ord=np.inf)),
        1.0,
    )
    if residual > 1e-7 * scale:
        raise ComputationError(
            "Submitted local update does not satisfy the local KKT stationarity condition.",
            {
                "worker_id": worker_id,
                "round": round_number,
                "stationarity_inf_residual": residual,
                "normalised_residual": residual / scale,
            },
        )


def consensus(x_new: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.asarray(x_new, dtype=float), axis=0)


def dual_updates(
    u_old: list[np.ndarray], x_new: list[np.ndarray], z_new: np.ndarray
) -> list[np.ndarray]:
    return [u + (x - z_new) for u, x in zip(u_old, x_new)]


def residuals(
    x_old: list[np.ndarray],
    x_new: list[np.ndarray],
    u_new: list[np.ndarray],
    z_old: np.ndarray,
    z_new: np.ndarray,
    rho: float,
) -> dict[str, object]:
    """Return consensus primal and coordinator dual residual vectors."""

    primal_vectors = np.asarray([x - z_new for x in x_new], dtype=float)
    # Every local block contributes the same z-change dual residual.
    dual_vectors = np.asarray([-rho * (z_new - z_old) for _ in x_new], dtype=float)
    x_stack = np.asarray(x_new, dtype=float)
    u_stack = np.asarray(u_new, dtype=float)
    return {
        "primal_vectors": primal_vectors,
        "dual_vectors": dual_vectors,
        "primal_residual": float(np.linalg.norm(primal_vectors)),
        "dual_residual": float(np.linalg.norm(dual_vectors)),
        "x_norm": float(np.linalg.norm(x_stack)),
        "z_norm": float(np.linalg.norm(z_new)),
        "dual_norm": float(rho * np.linalg.norm(u_stack)),
    }


def rescale_dual(u: list[np.ndarray], old_rho: float, new_rho: float) -> list[np.ndarray]:
    """Preserve the multiplier rho*u when changing rho."""

    ratio = old_rho / new_rho
    return [ratio * ui for ui in u]


def initial_state_values(problem: Problem) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    zero = np.zeros(problem.dimension, dtype=float)
    return [zero.copy() for _ in problem.models], zero.copy(), [zero.copy() for _ in problem.models]
