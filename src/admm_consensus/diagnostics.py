"""Independent numerical diagnostics and serialisable evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

import numpy as np

from .errors import ComputationError
from .models import Problem, SolverState


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:12]}"


def _array(value: np.ndarray | list[np.ndarray]) -> Any:
    if isinstance(value, list):
        return [np.asarray(v, dtype=float).tolist() for v in value]
    return np.asarray(value, dtype=float).tolist()


def centralized_solution(problem: Problem) -> dict[str, float | list[float]]:
    """Solve the aggregate KKT system independently of the ADMM iteration."""

    Q_total = sum(model.Q for model in problem.models)
    q_total = sum(model.q for model in problem.models)
    try:
        x_star = np.linalg.solve(Q_total, -q_total)
    except np.linalg.LinAlgError as exc:
        raise ComputationError("Centralized reference KKT matrix is singular.") from exc
    return {"x_star": x_star.tolist(), "objective_star": problem.total_objective(x_star)}


def state_snapshot(state: SolverState, *, include_problem: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_id": state.run_id,
        "status": state.status,
        "round_number": state.round_number,
        "rho": state.rho,
        "next_round_rho": state.public_rho_schedule(),
        "x": _array(state.x),
        "z": _array(state.z),
        "u": _array(state.u),
        "pending_worker_ids": sorted(state.pending),
        "trajectory": state.trajectory,
        "last_timeout": state.last_timeout,
        "failure_reason": state.failure_reason,
        "reference": centralized_solution(state.problem),
    }
    if include_problem:
        data["problem"] = {
            "dimension": state.problem.dimension,
            "workers": [
                {"worker_id": i, "Q": model.Q.tolist(), "q": model.q.tolist()}
                for i, model in enumerate(state.problem.models, start=1)
            ],
        }
        data["config"] = {
            "max_iterations": state.config.max_iterations,
            "rho": state.config.initial_rho,
            "tolerances": {
                "primal_abs": state.config.tol.primal_abs,
                "dual_abs": state.config.tol.dual_abs,
                "primal_rel": state.config.tol.primal_rel,
                "dual_rel": state.config.tol.dual_rel,
            },
            "adaptive_rho": None
            if state.config.adaptive_rho is None
            else {
                "mu": state.config.adaptive_rho.mu,
                "tau_incr": state.config.adaptive_rho.tau_incr,
                "tau_decr": state.config.adaptive_rho.tau_decr,
                "change_rounds": sorted(state.config.adaptive_rho.change_rounds),
            },
        }
        data["pending"] = {str(worker_id): x.tolist() for worker_id, x in state.pending.items()}
    return data


def task_payload(state: SolverState) -> dict[str, Any]:
    model = state.problem.models
    return {
        "run_id": state.run_id,
        "round_number": state.round_number,
        "rho": state.rho,
        "dimension": state.problem.dimension,
        "workers": [
            {
                "worker_id": i,
                "Q": model[i - 1].Q.tolist(),
                "q": model[i - 1].q.tolist(),
                "z": state.z.tolist(),
                "u": state.u[i - 1].tolist(),
            }
            for i in (1, 2)
        ],
        "submitted_worker_ids": sorted(state.pending),
        "status": state.status,
    }


def event_payload(state: SolverState, event_type: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": uuid.uuid4().hex,
        "timestamp": now_utc(),
        "run_id": state.run_id,
        "round_number": state.round_number,
        "rho": state.rho,
        "event_type": event_type,
        "details": details,
    }
