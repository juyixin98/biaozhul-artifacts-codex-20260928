"""Application services implementing the ADMM round state machine."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import ServerSettings
from .errors import (
    ComputationContractError,
    InputContractError,
    ModelParseError,
    NotFoundError,
    ResourceLimitError,
    StateConflictError,
)
from .evidence import digest, model_digest, round_token
from .kernel import (
    ADMMConfig,
    LocalUpdate,
    apply_round,
    check_convergence,
    effective_rho,
    parse_rho_schedule,
    rescale_dual,
)
from .models import QuadraticModel
from .parser import model_to_json, parse_problem
from .reference import central_solution
from .state import RoundState, RunState
from .storage import FileStore


def _now() -> float:
    return time.time()


class AdmmService:
    def __init__(self, store: FileStore, settings: ServerSettings) -> None:
        self.store = store
        self.settings = settings

    def create_run(self, payload: Mapping[str, Any]) -> RunState:
        if int(payload.get("dimension", 0)) > self.settings.max_problem_dim:
            raise ResourceLimitError(
                "problem dimension exceeds server limit",
                code="problem_dimension_limit_exceeded",
                details={
                    "dimension": payload.get("dimension"),
                    "max_dimension": self.settings.max_problem_dim,
                },
            )
        if int(payload.get("max_iterations", 0)) > self.settings.max_iterations:
            raise ResourceLimitError(
                "max_iterations exceeds server limit",
                code="max_iterations_limit_exceeded",
                details={
                    "max_iterations": payload.get("max_iterations"),
                    "server_max_iterations": self.settings.max_iterations,
                },
            )

        model = parse_problem(payload)
        if len(model.workers) != 2:
            raise InputContractError(
                "this teaching backend is scoped to exactly two compute processes",
                code="invalid_worker_count",
                details={"worker_count": len(model.workers)},
            )
        central = central_solution(model)
        if central.status not in {"unique_optimum", "minimum_norm_optimum"}:
            raise ModelParseError(
                "aggregated quadratic must have at least one finite minimizer",
                details={
                    "central_status": central.status,
                    "stationarity_residual": central.residual,
                },
            )

        n = model.dimension
        initial_z = payload.get("initial_z")
        if initial_z is None:
            z0 = np.zeros(n)
        else:
            z0 = np.asarray(initial_z, dtype=float)
            if z0.shape != (n,) or not np.all(np.isfinite(z0)):
                raise InputContractError(
                    f"initial_z must contain {n} finite values",
                    code="invalid_initial_z",
                )

        initial_rho = float(payload["initial_rho"])
        schedule = parse_rho_schedule(
            {
                int(item["iteration"]): float(item["rho"])
                for item in payload.get("rho_schedule", [])
            }
        )
        config = {
            "initial_rho": initial_rho,
            "eps_abs": float(payload["eps_abs"]),
            "eps_rel": float(payload["eps_rel"]),
            "max_iterations": int(payload["max_iterations"]),
            "deadline_ms": int(payload["deadline_ms"]),
            "rho_schedule": {
                str(iteration): rho
                for iteration, rho in sorted(schedule.items())
            },
        }
        admm_config = self._config_from_dict(config)
        run_id = uuid.uuid4().hex
        problem_json = model_to_json(model)
        problem_digest = model_digest(model)
        created = {
            "type": "run_created",
            "timestamp": _now(),
            "data": {
                "run_id": run_id,
                "name": str(payload["name"]),
                "problem": problem_json,
                "config": config,
                "problem_digest": problem_digest,
                "initial_rho": initial_rho,
                "initial_z": z0.tolist(),
                "central_reference": {
                    "status": central.status,
                    "objective": central.objective,
                    "x": central.x.tolist(),
                    "stationarity_residual": central.residual,
                    "min_eigenvalue": central.min_eigenvalue,
                },
            },
        }
        first_round = self._open_round_event(
            run_id=run_id,
            iteration=0,
            rho=effective_rho(admm_config, 0),
            rho_previous=initial_rho,
            z=z0,
            u={worker.worker_id: np.zeros(n) for worker in model.workers},
            deadline_ms=config["deadline_ms"],
            problem_digest=problem_digest,
        )
        return self.store.create_run(created, [first_round])

    def get_run(self, run_id: str) -> RunState:
        return self.store.load_run(run_id)

    def get_task(self, run_id: str, worker_id: str) -> dict[str, Any]:
        state = self.get_run(run_id)
        self._validate_worker(state, worker_id)
        if state.status not in {"running", "timed_out"}:
            return self._terminal_task_response(state, worker_id)
        round_state = state.current_round()
        now = _now()
        expired = now > round_state.deadline and round_state.status == "open"
        return {
            "run_id": run_id,
            "worker_id": worker_id,
            "status": "submitted" if worker_id in round_state.submissions else state.status,
            "iteration": round_state.iteration,
            "rho": round_state.rho,
            "z": round_state.z_pre,
            "u": round_state.u_pre[worker_id],
            "round_token": round_state.token,
            "deadline": round_state.deadline,
            "deadline_ms": round_state.deadline_ms,
            "already_submitted": worker_id in round_state.submissions,
            "expired": expired,
            "timeout_required": expired,
        }

    def submit(self, run_id: str, worker_id: str, payload: Mapping[str, Any]) -> RunState:
        with self.store.transaction(run_id) as (state, commit):
            self._validate_worker(state, worker_id)
            if state.status not in {"running", "timed_out"}:
                raise StateConflictError(
                    f"cannot submit after run status {state.status}",
                    code="run_not_active",
                    details={"status": state.status},
                )
            round_state = state.current_round()
            now = _now()
            if round_state.status == "open" and now > round_state.deadline:
                commit([self._timeout_event(
                        round_state,
                        "deadline_exceeded_before_submission",
                        now=now,
                    )],
                )
                raise StateConflictError(
                    "round deadline has expired; mark and resume before accepting work",
                    code="round_deadline_expired",
                )
            if round_state.status != "open":
                raise StateConflictError(
                    "round is timed out and must be resumed",
                    code="round_timed_out",
                )
            if worker_id in round_state.submissions:
                raise StateConflictError(
                    "worker already submitted for this round",
                    code="duplicate_submission",
                )

            model = parse_problem(state.problem)
            config = self._config(state)
            try:
                local = self._parse_submission(
                    model=model,
                    worker_id=worker_id,
                    round_state=round_state,
                    payload=payload,
                )
            except ComputationContractError:
                commit([self._rejected_event(round_state, worker_id, payload)],
                )
                raise

            events = [self._submission_event(round_state, worker_id, local, payload)]
            locals_for_round = {
                submitted_worker: LocalUpdate(
                    worker_id=submitted_worker,
                    iteration=round_state.iteration,
                    rho=round_state.rho,
                    z_pre=np.asarray(round_state.z_pre, dtype=float),
                    u_pre=np.asarray(
                        round_state.u_pre[submitted_worker], dtype=float
                    ),
                    x=np.asarray(submission.x, dtype=float),
                    objective=submission.objective,
                    stationarity_residual=submission.stationarity_residual,
                    solve_seconds=submission.solve_seconds,
                )
                for submitted_worker, submission in round_state.submissions.items()
            }
            locals_for_round[worker_id] = local
            if set(locals_for_round) == set(state.worker_ids):
                round_update = apply_round(
                    model=model,
                    iteration=round_state.iteration,
                    rho=round_state.rho,
                    rho_previous=round_state.rho_previous,
                    z_pre=np.asarray(round_state.z_pre, dtype=float),
                    u_pre={
                        key: np.asarray(value, dtype=float)
                        for key, value in round_state.u_pre.items()
                    },
                    locals_=locals_for_round,
                    config=config,
                )
                decision = check_convergence(round_update)
                is_last = decision.converged or (
                    round_state.iteration + 1 >= config.max_iterations
                )
                status = "converged" if decision.converged else (
                    "max_iterations_reached" if is_last else "running"
                )
                events.append({
                    "type": "round_completed",
                    "timestamp": _now(),
                    "data": self._completed_data(round_update, decision, status),
                })
                if not is_last:
                    events.append(
                        self._open_next_round_event(state, model, round_update, config)
                    )
            commit(events)
            return state

    def timeout_round(self, run_id: str, reason: str = "worker_deadline_exceeded") -> RunState:
        with self.store.transaction(run_id) as (state, commit):
            round_state = state.current_round()
            now = _now()
            if round_state.status != "open":
                raise StateConflictError(
                    f"round is already {round_state.status}",
                    code=f"round_{round_state.status}",
                )
            if now <= round_state.deadline:
                raise StateConflictError(
                    "round deadline has not elapsed",
                    code="deadline_not_elapsed",
                    details={"deadline": round_state.deadline, "now": now},
                )
            commit([self._timeout_event(round_state, reason, now=now)]
            )
            return state

    def resume_round(
        self,
        run_id: str,
        *,
        deadline_ms: int | None = None,
        reason: str = "resumed_after_timeout",
    ) -> RunState:
        with self.store.transaction(run_id) as (state, commit):
            round_state = state.current_round()
            if round_state.status != "timed_out":
                raise StateConflictError(
                    "only a timed-out round can be resumed",
                    code="round_not_timed_out",
                )
            new_deadline_ms = int(deadline_ms or round_state.deadline_ms)
            commit([{
                "type": "round_resumed",
                "timestamp": _now(),
                "data": {
                    "iteration": round_state.iteration,
                    "deadline": _now() + new_deadline_ms / 1000.0,
                    "deadline_ms": new_deadline_ms,
                    "retained_submissions": sorted(round_state.submissions),
                    "reason": reason,
                },
            }])
            return state

    @staticmethod
    def _validate_worker(state: RunState, worker_id: str) -> None:
        if worker_id not in state.worker_ids:
            raise NotFoundError(
                "worker is not part of this run",
                code="worker_not_found",
                details={
                    "worker_id": worker_id,
                    "known_workers": state.worker_ids,
                },
            )

    @staticmethod
    def _config_from_dict(config: Mapping[str, Any]) -> ADMMConfig:
        return ADMMConfig(
            initial_rho=float(config["initial_rho"]),
            eps_abs=float(config["eps_abs"]),
            eps_rel=float(config["eps_rel"]),
            max_iterations=int(config["max_iterations"]),
            deadline_ms=int(config["deadline_ms"]),
            rho_schedule={
                int(key): float(value)
                for key, value in config["rho_schedule"].items()
            },
        )

    def _config(self, state: RunState) -> ADMMConfig:
        return self._config_from_dict(state.config)

    def _open_round_event(
        self,
        *,
        run_id: str,
        iteration: int,
        rho: float,
        rho_previous: float,
        z: np.ndarray,
        u: Mapping[str, np.ndarray],
        deadline_ms: int,
        problem_digest: str,
    ) -> dict[str, Any]:
        z_list = np.asarray(z, dtype=float).tolist()
        u_array = {
            worker_id: rescale_dual(
                np.asarray(value, dtype=float), rho_previous, rho
            )
            for worker_id, value in u.items()
        }
        u_list = {
            worker_id: value.tolist() for worker_id, value in u_array.items()
        }
        token = round_token(
            run_id, iteration, rho, z_list, u_list, problem_digest
        )
        now = _now()
        return {
            "type": "round_opened",
            "timestamp": now,
            "data": {
                "iteration": iteration,
                "rho": float(rho),
                "rho_previous": float(rho_previous),
                "z_pre": z_list,
                "u_pre": u_list,
                "deadline_ms": int(deadline_ms),
                "deadline": now + deadline_ms / 1000.0,
                "round_token": token,
                "rho_changed": not np.isclose(rho, rho_previous),
                "dual_rescaled": not np.isclose(rho, rho_previous),
            },
        }

    def _open_next_round_event(
        self,
        state: RunState,
        model: QuadraticModel,
        round_update,
        config: ADMMConfig,
    ) -> dict[str, Any]:
        next_iteration = round_update.iteration + 1
        next_rho = effective_rho(config, next_iteration)
        return self._open_round_event(
            run_id=state.run_id,
            iteration=next_iteration,
            rho=next_rho,
            rho_previous=round_update.rho,
            z=round_update.z,
            u=round_update.u,
            deadline_ms=config.deadline_ms,
            problem_digest=state.problem_digest,
        )

    @staticmethod
    def _timeout_event(
        round_state: RoundState, reason: str, *, now: float
    ) -> dict[str, Any]:
        return {
            "type": "round_timed_out",
            "timestamp": now,
            "data": {
                "iteration": round_state.iteration,
                "reason": reason,
                "deadline": round_state.deadline,
                "submitted_workers": sorted(round_state.submissions),
                "missing_workers": sorted(
                    set(round_state.u_pre) - set(round_state.submissions)
                ),
            },
        }

    @staticmethod
    def _rejected_event(
        round_state: RoundState, worker_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "type": "submission_rejected",
            "timestamp": _now(),
            "data": {
                "iteration": round_state.iteration,
                "worker_id": worker_id,
                "reason": "computation_contract_failed",
                "payload_digest": digest(payload),
            },
        }

    @staticmethod
    def _submission_event(
        round_state: RoundState,
        worker_id: str,
        local: LocalUpdate,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "submission_received",
            "timestamp": _now(),
            "data": {
                "iteration": round_state.iteration,
                "worker_id": worker_id,
                "rho": local.rho,
                "z_pre": local.z_pre.tolist(),
                "u_pre": local.u_pre.tolist(),
                "x": local.x.tolist(),
                "objective": local.objective,
                "stationarity_residual": local.stationarity_residual,
                "solve_seconds": local.solve_seconds,
                "payload_digest": digest(payload),
            },
        }

    @staticmethod
    def _completed_data(round_update, decision, status: str) -> dict[str, Any]:
        return {
            "iteration": round_update.iteration,
            "rho": round_update.rho,
            "rho_previous": round_update.rho_previous,
            "z_pre": round_update.z_pre.tolist(),
            "z": round_update.z.tolist(),
            "x": {
                worker_id: value.tolist()
                for worker_id, value in round_update.x.items()
            },
            "u_pre": {
                worker_id: value.tolist()
                for worker_id, value in round_update.u_pre.items()
            },
            "u": {
                worker_id: value.tolist()
                for worker_id, value in round_update.u.items()
            },
            "primal_residual": round_update.primal_residual,
            "dual_residual": round_update.dual_residual,
            "eps_primal": round_update.eps_primal,
            "eps_dual": round_update.eps_dual,
            "primal_ok": decision.primal_ok,
            "dual_ok": decision.dual_ok,
            "converged": decision.converged,
            "reason": decision.reason,
            "status": status,
            "objective_total": round_update.objective_total,
            "objectives": round_update.objectives,
        }

    @staticmethod
    def _terminal_task_response(state: RunState, worker_id: str) -> dict[str, Any]:
        last = state.history[-1]
        return {
            "run_id": state.run_id,
            "worker_id": worker_id,
            "status": state.status,
            "iteration": state.iteration,
            "rho": state.rho,
            "z": state.z,
            "u": state.u[worker_id],
            "round_token": None,
            "deadline": None,
            "deadline_ms": 0,
            "already_submitted": True,
            "expired": False,
            "timeout_required": False,
            "end_reason": state.end_reason,
            "last_primal_ok": last.primal_ok,
            "last_dual_ok": last.dual_ok,
        }

    def _parse_submission(
        self,
        *,
        model: QuadraticModel,
        worker_id: str,
        round_state: RoundState,
        payload: Mapping[str, Any],
    ) -> LocalUpdate:
        if payload.get("round_token") != round_state.token:
            raise ComputationContractError(
                "submission round token does not match the open round",
                code="round_token_mismatch",
                details={
                    "expected": round_state.token,
                    "received": payload.get("round_token"),
                },
            )

        n = model.dimension
        values = {}
        for name in ("z_pre", "u_pre", "x"):
            try:
                array = np.asarray(payload.get(name), dtype=float)
            except (TypeError, ValueError) as exc:
                raise InputContractError(
                    f"{name} must be numeric", code=f"invalid_{name}"
                ) from exc
            if array.shape != (n,):
                raise InputContractError(
                    f"{name} must have length {n}",
                    code=f"invalid_{name}",
                    details={"actual_shape": list(array.shape)},
                )
            values[name] = array

        for name, array in values.items():
            if not np.all(np.isfinite(array)):
                raise ComputationContractError(
                    f"{name} contains a non-finite value",
                    code=f"non_finite_{name}",
                )

        try:
            rho = float(payload.get("rho"))
            objective = float(payload.get("objective"))
            stationarity = float(payload.get("stationarity_residual"))
            solve_seconds = float(payload.get("solve_seconds", 0.0))
        except (TypeError, ValueError) as exc:
            raise InputContractError("numeric scalar fields are invalid") from exc

        if not np.isfinite(rho) or rho <= 0:
            raise InputContractError("rho must be positive and finite")
        if not np.isclose(rho, round_state.rho, rtol=0.0, atol=1e-12):
            raise ComputationContractError(
                "submission used the wrong rho for this round",
                code="rho_mismatch",
                details={"expected": round_state.rho, "received": rho},
            )
        if not np.allclose(values["z_pre"], round_state.z_pre, rtol=0.0, atol=1e-10):
            raise ComputationContractError(
                "submission used a different prior consensus",
                code="z_pre_mismatch",
            )
        if not np.allclose(
            values["u_pre"],
            round_state.u_pre[worker_id],
            rtol=0.0,
            atol=1e-10,
        ):
            raise ComputationContractError(
                "submission used a different scaled dual",
                code="u_pre_mismatch",
            )
        if not np.isfinite(objective):
            raise ComputationContractError(
                "objective must be finite", code="non_finite_objective"
            )
        if not np.isfinite(stationarity) or stationarity < 0:
            raise InputContractError(
                "stationarity_residual must be a non-negative finite number"
            )
        if not np.isfinite(solve_seconds) or solve_seconds < 0:
            raise InputContractError("solve_seconds must be non-negative and finite")

        worker = model.worker(worker_id)
        x = values["x"]
        expected_objective = 0.5 * x @ worker.P @ x + worker.q @ x + worker.r
        if not np.isclose(
            objective,
            expected_objective,
            rtol=1e-7,
            atol=1e-7 * max(1.0, abs(float(expected_objective))),
        ):
            raise ComputationContractError(
                "reported objective does not match the submitted local point",
                code="objective_mismatch",
                details={
                    "reported": objective,
                    "recomputed": float(expected_objective),
                },
            )

        residual_vector = (
            worker.P @ x
            + worker.q
            + values["u_pre"]
            + rho * (x - values["z_pre"])
        )
        expected_stationarity = float(np.linalg.norm(residual_vector))
        tolerance = 1e-6 + 1e-6 * max(1.0, expected_stationarity)
        if stationarity > expected_stationarity + tolerance:
            raise ComputationContractError(
                "reported stationarity is inconsistent with submitted x",
                code="stationarity_mismatch",
                details={
                    "reported": stationarity,
                    "recomputed": expected_stationarity,
                },
            )

        return LocalUpdate(
            worker_id=worker_id,
            iteration=round_state.iteration,
            rho=rho,
            z_pre=values["z_pre"],
            u_pre=values["u_pre"],
            x=x,
            objective=objective,
            stationarity_residual=expected_stationarity,
            solve_seconds=solve_seconds,
        )
