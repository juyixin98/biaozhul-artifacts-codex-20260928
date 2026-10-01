"""FastAPI service entry."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import ServerSettings
from .errors import AppError
from .schemas import ResumeRequest, RunCreate, SubmissionCreate, TimeoutRequest
from .service import AdmmService
from .state import CompletedRound, RoundState, RunState
from .storage import FileStore


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    resolved_settings = settings or ServerSettings.from_env()
    store = FileStore(resolved_settings)
    service = AdmmService(store, resolved_settings)
    app = FastAPI(
        title="Local Two-Process Consensus ADMM",
        version="1.0.0",
        description=(
            "Teaching backend for separable convex quadratic objectives with "
            "durable rounds, residual diagnostics, and explicit timeouts."
        ),
    )
    app.state.settings = resolved_settings
    app.state.store = store
    app.state.service = service

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_response(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "category": "input_error",
                    "code": "request_validation_failed",
                    "message": "request does not satisfy the HTTP schema",
                    "details": {"errors": exc.errors(include_url=False)},
                }
            },
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "data_dir": str(resolved_settings.data_dir)}

    @app.post("/runs", status_code=201)
    def create_run(payload: RunCreate) -> dict[str, Any]:
        state = service.create_run(payload.model_dump())
        return serialize_run(state)

    @app.get("/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": [summarize_run(service.get_run(run_id)) for run_id in store.list_run_ids()]}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return serialize_run(service.get_run(run_id))

    @app.get("/runs/{run_id}/workers/{worker_id}/task")
    def get_task(run_id: str, worker_id: str) -> dict[str, Any]:
        return service.get_task(run_id, worker_id)

    @app.post("/runs/{run_id}/workers/{worker_id}/submissions")
    def submit(
        run_id: str, worker_id: str, payload: SubmissionCreate
    ) -> dict[str, Any]:
        state = service.submit(run_id, worker_id, payload.model_dump())
        return serialize_run(state)

    @app.post("/runs/{run_id}/timeout")
    def timeout_round(run_id: str, payload: TimeoutRequest) -> dict[str, Any]:
        state = service.timeout_round(run_id, payload.reason)
        return serialize_run(state)

    @app.post("/runs/{run_id}/resume")
    def resume_round(run_id: str, payload: ResumeRequest) -> dict[str, Any]:
        state = service.resume_round(
            run_id,
            deadline_ms=payload.deadline_ms,
            reason=payload.reason,
        )
        return serialize_run(state)

    @app.get("/runs/{run_id}/trajectory")
    def trajectory(run_id: str) -> dict[str, Any]:
        state = service.get_run(run_id)
        return {
            "run_id": run_id,
            "status": state.status,
            "iteration": state.iteration,
            "trajectory": [serialize_completed(item) for item in state.history],
        }

    @app.get("/runs/{run_id}/diagnostics")
    def diagnostics(run_id: str) -> dict[str, Any]:
        state = service.get_run(run_id)
        round_state = state.rounds.get(state.iteration) if state.status in {
            "running",
            "timed_out",
        } else None
        return {
            "run": summarize_run(state),
            "current_round": serialize_round(round_state) if round_state else None,
            "latest_completed_round": (
                serialize_completed(state.history[-1]) if state.history else None
            ),
            "failure_classification": classify_state(state),
        }

    @app.get("/runs/{run_id}/events")
    def events(run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "events": store.read_events(run_id)}

    return app


def classify_state(state: RunState) -> dict[str, Any]:
    if state.status == "converged":
        return {
            "category": "success",
            "code": "converged",
            "message": "both primal and dual tolerances are satisfied",
        }
    if state.status == "max_iterations_reached":
        last = state.history[-1]
        missing = []
        if not last.primal_ok:
            missing.append("primal")
        if not last.dual_ok:
            missing.append("dual")
        return {
            "category": "tolerance_not_reached",
            "code": "max_iterations_reached",
            "message": "iteration limit exhausted before both tolerances passed",
            "missing_conditions": missing,
            "primal_ok": last.primal_ok,
            "dual_ok": last.dual_ok,
        }
    if state.status == "timed_out":
        round_state = state.current_round()
        return {
            "category": "state_conflict",
            "code": "round_timed_out",
            "message": "a compute process did not submit before the round deadline",
            "submitted_workers": sorted(round_state.submissions),
            "missing_workers": sorted(
                set(round_state.u_pre) - set(round_state.submissions)
            ),
        }
    return {
        "category": "in_progress",
        "code": "running",
        "message": "round is open",
    }


def summarize_run(state: RunState) -> dict[str, Any]:
    last = state.history[-1] if state.history else None
    return {
        "run_id": state.run_id,
        "name": state.name,
        "status": state.status,
        "iteration": state.iteration,
        "rho": state.rho,
        "end_reason": state.end_reason,
        "problem_digest": state.problem_digest,
        "created_at": state.created_at,
        "latest_primal_residual": last.primal_residual if last else None,
        "latest_dual_residual": last.dual_residual if last else None,
        "latest_primal_ok": last.primal_ok if last else None,
        "latest_dual_ok": last.dual_ok if last else None,
        "latest_objective_total": last.objective_total if last else None,
    }


def serialize_round(round_state: RoundState) -> dict[str, Any]:
    return {
        "iteration": round_state.iteration,
        "status": round_state.status,
        "rho": round_state.rho,
        "rho_previous": round_state.rho_previous,
        "deadline": round_state.deadline,
        "deadline_ms": round_state.deadline_ms,
        "round_token": round_state.token,
        "z_pre": round_state.z_pre,
        "u_pre": round_state.u_pre,
        "submitted_workers": sorted(round_state.submissions),
        "missing_workers": sorted(
            set(round_state.u_pre) - set(round_state.submissions)
        ),
        "submissions": {
            worker_id: {
                "x": submission.x,
                "objective": submission.objective,
                "stationarity_residual": submission.stationarity_residual,
                "solve_seconds": submission.solve_seconds,
            }
            for worker_id, submission in round_state.submissions.items()
        },
        "timeout_reason": round_state.timeout_reason,
    }


def serialize_completed(item: CompletedRound) -> dict[str, Any]:
    return {
        "iteration": item.iteration,
        "rho": item.rho,
        "rho_previous": item.rho_previous,
        "z": item.z,
        "x": item.x,
        "u": item.u,
        "primal_residual": item.primal_residual,
        "dual_residual": item.dual_residual,
        "eps_primal": item.eps_primal,
        "eps_dual": item.eps_dual,
        "primal_ok": item.primal_ok,
        "dual_ok": item.dual_ok,
        "converged": item.converged,
        "reason": item.reason,
        "objective_total": item.objective_total,
        "objectives": item.objectives,
    }


def serialize_run(state: RunState) -> dict[str, Any]:
    result = summarize_run(state)
    result.update(
        {
            "problem": state.problem,
            "config": state.config,
            "central_reference": state.central_reference,
            "z": state.z,
            "u": state.u,
            "current_round": (
                serialize_round(state.rounds[state.iteration])
                if state.status in {"running", "timed_out"}
                else None
            ),
            "history": [serialize_completed(item) for item in state.history],
            "failure_classification": classify_state(state),
        }
    )
    return result


app = create_app()
