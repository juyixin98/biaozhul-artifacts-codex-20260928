"""HTTP service entry point for the local consensus ADMM backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .coordinator import Coordinator
from .errors import ADMMError, InputError
from .parser import parse_config, parse_problem
from .repository import RunRepository
from .schemas import CreateRunRequest, SubmitRequest, TimeoutRequest


settings = Settings.from_env()
repository = RunRepository(settings)
coordinator = Coordinator(repository)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.repository = repository
    app.state.coordinator = coordinator
    yield


app = FastAPI(
    title="Consensus ADMM Teaching Backend",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ADMMError)
async def admm_error_handler(request: Request, exc: ADMMError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "data_dir": str(settings.data_dir),
        "max_dimension": settings.max_dimension,
        "max_iterations": settings.max_iterations,
        "max_runs": settings.max_runs,
    }


@app.post("/runs", status_code=201)
def create_run(request: CreateRunRequest) -> dict[str, Any]:
    with repository.lock:
        payload = request.model_dump()
        problem = parse_problem(payload, settings)
        config = parse_config(payload.get("config", {}), settings)
        state = coordinator.create_run(problem, config)
        return {"run": coordinator.snapshot(state)}


@app.get("/runs")
def list_runs() -> dict[str, Any]:
    with repository.lock:
        return {"run_ids": repository.list_runs()}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with repository.lock:
        state = repository.load(run_id)
        return {"run": coordinator.snapshot(state)}


@app.get("/runs/{run_id}/events")
def get_events(run_id: str) -> dict[str, Any]:
    with repository.lock:
        return {"events": repository.events(run_id)}


@app.get("/runs/{run_id}/task")
def get_task(run_id: str, worker_id: int | None = None) -> dict[str, Any]:
    with repository.lock:
        state = repository.load(run_id)
        if worker_id is not None and worker_id not in (1, 2):
            raise InputError("worker_id must be 1 or 2.", {"worker_id": worker_id})
        from .diagnostics import task_payload

        return {"task": task_payload(state)}


@app.post("/runs/{run_id}/submit")
def submit_update(run_id: str, request: SubmitRequest) -> dict[str, Any]:
    with repository.lock:
        state = repository.load(run_id)
        x = np.asarray(request.x, dtype=float)
        coordinator.submit(
            state,
            request.worker_id,
            x,
            round_number=request.round_number,
            rho=request.rho,
        )
        return {"run": coordinator.snapshot(state)}


@app.post("/runs/{run_id}/complete")
def complete_round(run_id: str) -> dict[str, Any]:
    with repository.lock:
        state = repository.load(run_id)
        coordinator.complete_round(state)
        return {"run": coordinator.snapshot(state)}


@app.post("/runs/{run_id}/timeout")
def timeout_round(run_id: str, request: TimeoutRequest) -> dict[str, Any]:
    with repository.lock:
        state = repository.load(run_id)
        coordinator.timeout_round(
            state,
            missing_worker_ids=request.missing_worker_ids,
            reason=request.reason,
        )
        return {"run": coordinator.snapshot(state)}


@app.post("/runs/{run_id}/auto-round")
def auto_round(run_id: str) -> dict[str, Any]:
    with repository.lock:
        state = repository.load(run_id)
        coordinator.auto_round(state)
        return {"run": coordinator.snapshot(state)}
