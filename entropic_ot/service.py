"""FastAPI entry point for entropy-regularized optimal transport."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .diagnostics import RunRecorder, new_run_id
from .errors import OTError, ResourceExhaustedError, StateConflictError
from .kernel import solve_entropic_ot
from .models import SolveRequest, SolveResponse, serialize_result
from .state import IdempotencyStore, StoredResponse


def create_app(log_dir: str | os.PathLike[str] | None = None) -> FastAPI:
    resolved_log_dir = str(
        log_dir or os.environ.get("ENTROPIC_OT_LOG_DIR", "runs")
    )
    app = FastAPI(
        title="Entropy-regularized Optimal Transport",
        version="1.0.0",
    )
    app.state.log_dir = resolved_log_dir
    app.state.idempotency = IdempotencyStore()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/solve", response_model=SolveResponse)
    def solve(request: SolveRequest) -> JSONResponse:
        store: IdempotencyStore = app.state.idempotency
        solver_payload = _solver_payload(request)
        payload_hash = store.payload_hash(solver_payload)
        key = request.idempotency_key
        if key is not None:
            stored = store.get_or_lock(key, payload_hash)
            if stored is not None:
                return JSONResponse(
                    status_code=stored.status_code,
                    content=_mark_replay(stored.body),
                )

        run_id = new_run_id()
        recorder = RunRecorder(resolved_log_dir, run_id=run_id)
        try:
            result = solve_entropic_ot(
                request.a,
                request.b,
                request.cost,
                request.epsilon,
                max_iterations=request.max_iterations,
                tolerance=request.tolerance,
                recorder=recorder,
                include_logged_inputs=True,
            )
            body = serialize_result(run_id, result, request_hash=payload_hash)
            if result.stop_reason == "iteration_budget_exhausted" and not result.feasible:
                error = ResourceExhaustedError(
                    "Iteration budget expired before both marginals were satisfied.",
                    details={
                        "best_effort": body,
                        "row_error": result.row_error,
                        "column_error": result.column_error,
                    },
                    run_id=run_id,
                )
                recorder.event(
                    "failure",
                    category=error.category,
                    message=error.message,
                    details=error.details,
                )
                response = JSONResponse(status_code=error.status_code, content=error.to_dict())
                if key is not None:
                    store.store(
                        key,
                        payload_hash,
                        StoredResponse(error.status_code, error.to_dict()),
                    )
                return response

            response = JSONResponse(status_code=200, content=body)
            if key is not None:
                store.store(key, payload_hash, StoredResponse(200, body))
            return response
        except Exception:
            if key is not None:
                store.release(key, payload_hash)
            raise

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "category": "input_quality",
                    "code": "RequestValidationError",
                    "message": "Request model validation failed.",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                    "run_id": None,
                }
            },
        )

    @app.exception_handler(OTError)
    async def ot_error_handler(request: Request, exc: OTError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    return app


def _solver_payload(request: SolveRequest) -> dict[str, Any]:
    return {
        "a": request.a,
        "b": request.b,
        "cost": request.cost,
        "epsilon": request.epsilon,
        "max_iterations": request.max_iterations,
        "tolerance": request.tolerance,
    }


def _mark_replay(body: dict[str, Any]) -> dict[str, Any]:
    if "replay" in body:
        body["replay"]["replayed"] = True
    return body


app = create_app()
