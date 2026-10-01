"""State machine and event reducer for ADMM runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import StateConflictError


@dataclass
class SubmissionState:
    worker_id: str
    iteration: int
    rho: float
    x: list[float]
    objective: float
    stationarity_residual: float
    solve_seconds: float


@dataclass
class RoundState:
    iteration: int
    rho: float
    rho_previous: float
    opened_at: float
    deadline: float
    deadline_ms: int
    token: str
    z_pre: list[float]
    u_pre: dict[str, list[float]]
    submissions: dict[str, SubmissionState] = field(default_factory=dict)
    status: str = "open"
    timeout_at: float | None = None
    timeout_reason: str | None = None
    resumed_at: float | None = None


@dataclass
class CompletedRound:
    iteration: int
    rho: float
    rho_previous: float
    z_pre: list[float]
    z: list[float]
    x: dict[str, list[float]]
    u_pre: dict[str, list[float]]
    u: dict[str, list[float]]
    primal_residual: float
    dual_residual: float
    eps_primal: float
    eps_dual: float
    primal_ok: bool
    dual_ok: bool
    converged: bool
    reason: str
    objective_total: float
    objectives: dict[str, float]


@dataclass
class RunState:
    run_id: str
    name: str
    problem: dict[str, Any]
    config: dict[str, Any]
    problem_digest: str
    central_reference: dict[str, Any] | None = None
    status: str = "running"
    iteration: int = 0
    rho: float = 1.0
    z: list[float] = field(default_factory=list)
    u: dict[str, list[float]] = field(default_factory=dict)
    rounds: dict[int, RoundState] = field(default_factory=dict)
    history: list[CompletedRound] = field(default_factory=list)
    end_reason: str | None = None
    last_event_id: int = 0
    created_at: float | None = None

    @property
    def worker_ids(self) -> list[str]:
        return [worker["id"] for worker in self.problem["workers"]]

    @property
    def dimension(self) -> int:
        return int(self.problem["dimension"])

    def current_round(self) -> RoundState:
        if self.status not in {"running", "timed_out"}:
            raise StateConflictError(
                f"run is {self.status}", code="run_not_active"
            )
        if self.iteration not in self.rounds:
            raise StateConflictError("current round is missing", code="round_missing")
        return self.rounds[self.iteration]


def _round_from_event(event: dict[str, Any]) -> RoundState:
    data = event["data"]
    return RoundState(
        iteration=int(data["iteration"]),
        rho=float(data["rho"]),
        rho_previous=float(data["rho_previous"]),
        opened_at=float(event["timestamp"]),
        deadline=float(data["deadline"]),
        deadline_ms=int(data["deadline_ms"]),
        token=str(data["round_token"]),
        z_pre=list(data["z_pre"]),
        u_pre={key: list(value) for key, value in data["u_pre"].items()},
    )


def _submission_from_event(event: dict[str, Any]) -> SubmissionState:
    data = event["data"]
    return SubmissionState(
        worker_id=str(data["worker_id"]),
        iteration=int(data["iteration"]),
        rho=float(data["rho"]),
        x=list(data["x"]),
        objective=float(data["objective"]),
        stationarity_residual=float(data["stationarity_residual"]),
        solve_seconds=float(data.get("solve_seconds", 0.0)),
    )


def _completed_from_event(event: dict[str, Any]) -> CompletedRound:
    data = event["data"]
    return CompletedRound(
        iteration=int(data["iteration"]),
        rho=float(data["rho"]),
        rho_previous=float(data["rho_previous"]),
        z_pre=list(data["z_pre"]),
        z=list(data["z"]),
        x={key: list(value) for key, value in data["x"].items()},
        u_pre={key: list(value) for key, value in data["u_pre"].items()},
        u={key: list(value) for key, value in data["u"].items()},
        primal_residual=float(data["primal_residual"]),
        dual_residual=float(data["dual_residual"]),
        eps_primal=float(data["eps_primal"]),
        eps_dual=float(data["eps_dual"]),
        primal_ok=bool(data["primal_ok"]),
        dual_ok=bool(data["dual_ok"]),
        converged=bool(data["converged"]),
        reason=str(data["reason"]),
        objective_total=float(data["objective_total"]),
        objectives={key: float(value) for key, value in data["objectives"].items()},
    )


def initial_state_from_created(event: dict[str, Any]) -> RunState:
    data = event["data"]
    state = RunState(
        run_id=str(data["run_id"]),
        name=str(data["name"]),
        problem=dict(data["problem"]),
        config=dict(data["config"]),
        problem_digest=str(data["problem_digest"]),
        central_reference=data.get("central_reference"),
        created_at=float(event["timestamp"]),
        rho=float(data["initial_rho"]),
        z=list(data["initial_z"]),
        u={
            worker["id"]: [0.0] * int(data["problem"]["dimension"])
            for worker in data["problem"]["workers"]
        },
    )
    state.last_event_id = int(event["seq"])
    return state


def apply_event(state: RunState, event: dict[str, Any]) -> RunState:
    kind = event["type"]
    data = event["data"]
    state.last_event_id = int(event["seq"])

    if kind == "run_created":
        raise StateConflictError("run_created cannot be replayed onto a run")
    if kind == "round_opened":
        if state.rounds.get(int(data["iteration"])) is not None:
            raise StateConflictError("duplicate round opening")
        state.rounds[int(data["iteration"])] = _round_from_event(event)
        state.status = "running"
        state.rho = float(data["rho"])
    elif kind == "submission_received":
        round_state = state.rounds[int(data["iteration"])]
        worker_id = str(data["worker_id"])
        if worker_id in round_state.submissions:
            raise StateConflictError(
                "duplicate worker submission", code="duplicate_submission"
            )
        round_state.submissions[worker_id] = _submission_from_event(event)
    elif kind == "submission_rejected":
        # Evidence-only event: the state is unchanged after a failed submission.
        pass
    elif kind == "round_timed_out":
        round_state = state.rounds[int(data["iteration"])]
        round_state.status = "timed_out"
        round_state.timeout_at = float(event["timestamp"])
        round_state.timeout_reason = str(data.get("reason", "timeout"))
        state.status = "timed_out"
    elif kind == "round_resumed":
        round_state = state.rounds[int(data["iteration"])]
        round_state.status = "open"
        round_state.deadline = float(data["deadline"])
        if data.get("deadline_ms") is not None:
            round_state.deadline_ms = int(data["deadline_ms"])
        round_state.resumed_at = float(event["timestamp"])
        round_state.timeout_at = None
        round_state.timeout_reason = None
        state.status = "running"
    elif kind == "round_completed":
        completed = _completed_from_event(event)
        round_state = state.rounds[completed.iteration]
        round_state.status = "completed"
        state.history.append(completed)
        state.iteration = completed.iteration + 1
        state.z = completed.z
        state.u = completed.u
        state.rho = completed.rho
        state.status = str(data["status"])
        state.end_reason = data.get("reason")
    else:
        raise StateConflictError(f"unknown event type {kind}", code="unknown_event")
    return state
