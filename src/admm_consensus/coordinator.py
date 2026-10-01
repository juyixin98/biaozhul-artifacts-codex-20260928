"""Pure ADMM coordination logic and durable state transitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .diagnostics import centralized_solution, new_run_id, state_snapshot
from .errors import ComputationError, InputError, StateConflict
from .kernel import (
    consensus,
    dual_updates,
    initial_state_values,
    rescale_dual,
    residuals,
    solve_local_update,
    validate_local_update,
)
from .models import Problem, SolverConfig, SolverState
from .repository import RunRepository


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


class Coordinator:
    def __init__(self, repository: RunRepository) -> None:
        self.repository = repository

    def create_run(self, problem: Problem, config: SolverConfig) -> SolverState:
        x, z, u = initial_state_values(problem)
        state = SolverState(
            run_id=new_run_id(),
            problem=problem,
            config=config,
            status="running",
            round_number=1,
            rho=config.initial_rho,
            x=x,
            z=z,
            u=u,
            pending={},
            trajectory=[],
        )
        reference = centralized_solution(problem)
        self.repository.create(state)
        self.repository.append_event(
            state,
            "initial_task",
            {
                "phase_order": ["local_updates", "consensus", "dual", "residuals", "termination", "rho_schedule"],
                "dimension": problem.dimension,
                "reference": reference,
            },
        )
        self.repository.save(state)
        return state

    def get(self, run_id: str) -> SolverState:
        return self.repository.load(run_id)

    def submit(self, state: SolverState, worker_id: int, x: np.ndarray, *, round_number: int, rho: float) -> SolverState:
        if worker_id not in (1, 2):
            raise InputError("worker_id must be 1 or 2.", {"worker_id": worker_id})
        try:
            if state.status != "running":
                raise StateConflict(
                    "Run is not accepting submissions.",
                    {"run_id": state.run_id, "status": state.status},
                )
            if worker_id in state.pending:
                raise StateConflict(
                    "Worker has already submitted this round.",
                    {"worker_id": worker_id, "round": state.round_number},
                )
            validate_local_update(state, worker_id, x, round_number=round_number, rho=rho)
        except Exception as exc:
            self.repository.append_event(
                state,
                "submission_rejected",
                {
                    "worker_id": worker_id,
                    "submitted_round": round_number,
                    "submitted_rho": rho,
                    "error_code": getattr(exc, "code", "computation"),
                    "message": str(exc),
                    "details": getattr(exc, "details", {}),
                },
            )
            raise
        state.pending[worker_id] = x.copy()
        self.repository.append_event(
            state,
            "local_submitted",
            {
                "phase": "local_update",
                "worker_id": worker_id,
                "round_number": state.round_number,
                "rho": state.rho,
                "x": x.tolist(),
                "accepted_worker_ids": sorted(state.pending),
            },
        )
        self.repository.save(state)
        return state

    def timeout_round(self, state: SolverState, *, missing_worker_ids: list[int] | None = None, reason: str = "deadline_exceeded") -> SolverState:
        if state.status != "running":
            raise StateConflict("Only running rounds can time out.", {"status": state.status})
        missing = sorted(set(missing_worker_ids or [worker_id for worker_id in (1, 2) if worker_id not in state.pending]))
        invalid = [worker_id for worker_id in missing if worker_id not in (1, 2)]
        if invalid:
            raise InputError("Missing worker IDs must be 1 or 2.", {"worker_ids": invalid})
        if not missing:
            raise StateConflict("Both workers have submitted; this round cannot be marked incomplete.")
        provisional = sorted(state.pending)
        state.pending.clear()
        state.last_timeout = {
            "round_number": state.round_number,
            "missing_worker_ids": missing,
            "discarded_worker_ids": provisional,
            "reason": reason,
        }
        self.repository.append_event(
            state,
            "round_timeout",
            {
                "round_number": state.round_number,
                "rho": state.rho,
                "missing_worker_ids": missing,
                "discarded_worker_ids": provisional,
                "reason": reason,
                "decision": "whole_round_rejected;_same_round_remains_open",
            },
        )
        self.repository.save(state)
        return state

    def auto_round(self, state: SolverState) -> SolverState:
        """Run the fixed local step in-process for both worker processes."""

        if state.pending:
            raise StateConflict(
                "Manual submissions are pending; finish or time out that round first.",
                {"pending_worker_ids": sorted(state.pending)},
            )
        submitted: list[np.ndarray] = []
        try:
            for worker_id, model in enumerate(state.problem.models, start=1):
                x_i = solve_local_update(
                    model,
                    z=state.z,
                    u=state.u[worker_id - 1],
                    rho=state.rho,
                    worker_id=worker_id,
                )
                submitted.append(x_i)
                self.repository.append_event(
                    state,
                    "local_submitted",
                    {
                        "phase": "local_update",
                        "worker_id": worker_id,
                        "round_number": state.round_number,
                        "rho": state.rho,
                        "x": x_i.tolist(),
                        "source": "local_kernel",
                    },
                )
        except ComputationError as exc:
            state.status = "failed"
            state.failure_reason = exc.message
            self.repository.append_event(
                state,
                "computation_failed",
                {"phase": "local_update", "message": exc.message, "details": exc.details},
            )
            self.repository.save(state)
            raise
        state.pending = {worker_id: x.copy() for worker_id, x in enumerate(submitted, start=1)}
        return self._complete_round(state)

    def complete_round(self, state: SolverState) -> SolverState:
        if set(state.pending) != {1, 2}:
            raise StateConflict(
                "Consensus requires both submissions for the same round.",
                {
                    "round": state.round_number,
                    "submitted_worker_ids": sorted(state.pending),
                    "missing_worker_ids": [i for i in (1, 2) if i not in state.pending],
                },
            )
        return self._complete_round(state)

    def _complete_round(self, state: SolverState) -> SolverState:
        completed_round = state.round_number
        x_old = [x.copy() for x in state.x]
        z_old = state.z.copy()
        x_new = [state.pending[worker_id].copy() for worker_id in (1, 2)]

        z_new = consensus(x_new)
        u_new = dual_updates(state.u, x_new, z_new)
        metrics = residuals(x_old, x_new, u_new, z_old, z_new, state.rho)
        primal_threshold = state.config.tol.primal_threshold(float(metrics["x_norm"]), float(metrics["z_norm"]))
        dual_threshold = state.config.tol.dual_threshold(float(metrics["dual_norm"]))
        primal_ok = float(metrics["primal_residual"]) <= primal_threshold
        dual_ok = float(metrics["dual_residual"]) <= dual_threshold
        converged = primal_ok and dual_ok
        objective = state.problem.total_objective(z_new)

        state.x = x_new
        state.z = z_new
        state.u = u_new
        state.pending.clear()

        record: dict[str, Any] = {
            "round": completed_round,
            "rho": state.rho,
            "phase_order": ["local_updates", "consensus", "dual", "residuals", "termination"],
            "x_old": [v.tolist() for v in x_old],
            "x": [v.tolist() for v in x_new],
            "z": z_new.tolist(),
            "u": [v.tolist() for v in u_new],
            "primal_vectors": _jsonable(metrics["primal_vectors"]),
            "dual_vectors": _jsonable(metrics["dual_vectors"]),
            "primal_residual": float(metrics["primal_residual"]),
            "dual_residual": float(metrics["dual_residual"]),
            "primal_threshold": primal_threshold,
            "dual_threshold": dual_threshold,
            "primal_ok": primal_ok,
            "dual_ok": dual_ok,
            "objective": objective,
        }
        state.trajectory.append(record)

        if converged:
            state.status = "converged"
            state.round_number = completed_round
            termination = "converged: primal and dual residual thresholds both satisfied"
        elif completed_round >= state.config.max_iterations:
            state.status = "max_iterations"
            state.round_number = completed_round
            termination = "not converged: max_iterations reached"
        else:
            state.round_number = completed_round + 1
            termination = "continue"
            self._schedule_rho(state, completed_round, float(metrics["primal_residual"]), float(metrics["dual_residual"]))

        record["termination"] = termination
        self.repository.append_event(state, "round_completed", _jsonable(record))
        self.repository.save(state)
        return state

    def _schedule_rho(self, state: SolverState, completed_round: int, primal_residual: float, dual_residual: float) -> None:
        adaptive = state.config.adaptive_rho
        if adaptive is None:
            return
        old_rho = state.rho
        new_rho = adaptive.next_rho(
            round_number=completed_round,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            rho=old_rho,
        )
        if np.isclose(new_rho, old_rho, rtol=0.0, atol=1e-15):
            return
        state.u = rescale_dual(state.u, old_rho, new_rho)
        state.rho = float(new_rho)
        self.repository.append_event(
            state,
            "rho_changed",
            {
                "after_round": completed_round,
                "old_rho": old_rho,
                "new_rho": new_rho,
                "dual_scale": old_rho / new_rho,
                "reason": {
                    "primal_residual": primal_residual,
                    "dual_residual": dual_residual,
                    "mu": adaptive.mu,
                    "tau_incr": adaptive.tau_incr,
                    "tau_decr": adaptive.tau_decr,
                },
                "u_scaled": [v.tolist() for v in state.u],
            },
        )

    def snapshot(self, state: SolverState) -> dict[str, Any]:
        return state_snapshot(state)
