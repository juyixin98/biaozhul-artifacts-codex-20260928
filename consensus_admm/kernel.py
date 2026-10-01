"""Pure ADMM numeric operations for separable convex quadratics."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .errors import KernelComputationError
from .models import (
    ADMMConfig,
    ConvergenceDecision,
    LocalUpdate,
    QuadraticModel,
    QuadraticWorker,
    RoundUpdate,
)


def vector(value: Sequence[float] | np.ndarray, name: str, n: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (n,):
        raise KernelComputationError(
            f"{name} must have shape ({n},)",
            details={"actual_shape": list(array.shape)},
        )
    if not np.all(np.isfinite(array)):
        raise KernelComputationError(f"{name} contains non-finite values")
    return array


def parse_rho_schedule(
    schedule: Sequence[tuple[int, float]] | Mapping[int, float],
) -> dict[int, float]:
    result = dict(schedule)
    for iteration, rho in result.items():
        if not isinstance(iteration, int) or iteration < 0:
            raise KernelComputationError(
                "rho schedule iteration must be a non-negative integer"
            )
        if not np.isfinite(rho) or rho <= 0:
            raise KernelComputationError(
                "rho schedule values must be positive finite numbers"
            )
    return result


def effective_rho(config: ADMMConfig, iteration: int) -> float:
    return float(config.rho_schedule.get(iteration, config.initial_rho))


def rescale_dual(u: np.ndarray, old_rho: float, new_rho: float) -> np.ndarray:
    """Preserve the unscaled dual y=rho*u when rho changes."""
    if old_rho <= 0 or new_rho <= 0:
        raise KernelComputationError("rho values must be positive")
    return np.asarray(u, dtype=float) * (old_rho / new_rho)


def begin_round_dual(
    u: Mapping[str, np.ndarray], old_rho: float, new_rho: float
) -> dict[str, np.ndarray]:
    return {
        worker_id: rescale_dual(np.asarray(value, dtype=float), old_rho, new_rho)
        for worker_id, value in u.items()
    }


def quadratic_objective(worker: QuadraticWorker, x: np.ndarray) -> float:
    return float(0.5 * x @ worker.P @ x + worker.q @ x + worker.r)


def solve_local(
    worker: QuadraticWorker,
    iteration: int,
    rho: float,
    z_pre: np.ndarray,
    u_pre: np.ndarray,
) -> LocalUpdate:
    """Use only z/u from the current round for the local-variable update."""
    if rho <= 0:
        raise KernelComputationError("rho must be positive")
    n = worker.P.shape[0]
    z = vector(z_pre, "z_pre", n)
    u = vector(u_pre, "u_pre", n)

    started = time.perf_counter()
    system = worker.P + rho * np.eye(n)
    rhs = rho * z - u - worker.q
    try:
        factorization = cho_factor(system)
        x = cho_solve(factorization, rhs)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise KernelComputationError(
            "local quadratic solve failed",
            details={"worker_id": worker.worker_id},
        ) from exc
    solve_seconds = time.perf_counter() - started

    x = vector(x, "x", n)
    stationarity = worker.P @ x + worker.q + u + rho * (x - z)
    return LocalUpdate(
        worker_id=worker.worker_id,
        iteration=iteration,
        rho=float(rho),
        z_pre=z,
        u_pre=u,
        x=x,
        objective=quadratic_objective(worker, x),
        stationarity_residual=float(np.linalg.norm(stationarity)),
        solve_seconds=float(solve_seconds),
    )


def apply_round(
    model: QuadraticModel,
    iteration: int,
    rho: float,
    rho_previous: float,
    z_pre: np.ndarray,
    u_pre: Mapping[str, np.ndarray],
    locals_: Mapping[str, LocalUpdate],
    config: ADMMConfig,
) -> RoundUpdate:
    """Apply z then dual updates after every local update has been received."""
    if rho <= 0 or rho_previous <= 0:
        raise KernelComputationError("rho values must be positive")
    n = model.dimension
    old_z = vector(z_pre, "z_pre", n)
    if set(locals_) != set(model.worker_ids):
        raise KernelComputationError(
            "a round requires exactly one submission from every configured worker",
            details={
                "expected": list(model.worker_ids),
                "received": sorted(locals_),
            },
        )

    x: dict[str, np.ndarray] = {}
    objectives: dict[str, float] = {}
    start_dual = {
        worker_id: vector(value, "u_pre", n)
        for worker_id, value in u_pre.items()
    }
    for worker in model.workers:
        local = locals_[worker.worker_id]
        if local.iteration != iteration:
            raise KernelComputationError(
                "local update belongs to a different iteration",
                details={
                    "worker_id": worker.worker_id,
                    "expected_iteration": iteration,
                    "actual_iteration": local.iteration,
                },
            )
        if not np.isclose(local.rho, rho, rtol=1e-12, atol=1e-12):
            raise KernelComputationError(
                "local update used a different rho",
                details={
                    "worker_id": worker.worker_id,
                    "expected_rho": rho,
                    "actual_rho": local.rho,
                },
            )
        local_z = vector(local.z_pre, "local.z_pre", n)
        local_u = vector(local.u_pre, "local.u_pre", n)
        if not np.allclose(local_z, old_z, rtol=0.0, atol=1e-12):
            raise KernelComputationError(
                "local update used a different prior consensus",
                details={"worker_id": worker.worker_id},
            )
        if not np.allclose(
            local_u, start_dual[worker.worker_id], rtol=0.0, atol=1e-12
        ):
            raise KernelComputationError(
                "local update used a different prior dual",
                details={"worker_id": worker.worker_id},
            )
        x[worker.worker_id] = vector(local.x, "local.x", n)
        objectives[worker.worker_id] = float(local.objective)

    # Fixed order: all x updates, consensus z, then scaled dual u.
    z = np.mean(np.stack([x[worker_id] for worker_id in model.worker_ids]), axis=0)
    u = {
        worker_id: start_dual[worker_id] + x[worker_id] - z
        for worker_id in model.worker_ids
    }

    count = len(model.workers)
    x_stack = np.concatenate([x[worker_id] for worker_id in model.worker_ids])
    z_stack = np.tile(z, count)
    y_stack = np.concatenate(
        [rho * u[worker_id] for worker_id in model.worker_ids]
    )
    primal_residual = float(np.linalg.norm(x_stack - z_stack))
    dual_residual = float(np.sqrt(count) * rho * np.linalg.norm(z - old_z))
    eps_primal = float(
        np.sqrt(n * count) * config.eps_abs
        + config.eps_rel * max(np.linalg.norm(x_stack), np.linalg.norm(z_stack))
    )
    eps_dual = float(
        np.sqrt(n * count) * config.eps_abs
        + config.eps_rel * np.linalg.norm(y_stack)
    )

    return RoundUpdate(
        iteration=iteration,
        rho=float(rho),
        rho_previous=float(rho_previous),
        z_pre=old_z,
        z=z,
        x=x,
        u_pre=start_dual,
        u=u,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        eps_primal=eps_primal,
        eps_dual=eps_dual,
        objectives=objectives,
        objective_total=float(sum(objectives.values())),
    )


def check_convergence(round_update: RoundUpdate) -> ConvergenceDecision:
    """Primal and dual termination are separate boolean conditions."""
    primal_ok = round_update.primal_residual <= round_update.eps_primal
    dual_ok = round_update.dual_residual <= round_update.eps_dual
    if primal_ok and dual_ok:
        return ConvergenceDecision(
            True,
            True,
            True,
            "primal_and_dual_tolerances_satisfied",
        )
    missing = []
    if not primal_ok:
        missing.append("primal")
    if not dual_ok:
        missing.append("dual")
    return ConvergenceDecision(
        False,
        primal_ok,
        dual_ok,
        "tolerance_not_satisfied:" + "_and_".join(missing),
    )
