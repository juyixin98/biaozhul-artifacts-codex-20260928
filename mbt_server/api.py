"""FastAPI report and test-execution interface."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .core.domain import RunRequest, StateMachineModel
from .core.engine import run_model_test
from .core.errors import ErrorCategory, ModelValidationError
from .core.storage import EvidenceStore


def create_app(database_path: str | None = None) -> FastAPI:
    database_path = database_path or os.environ.get(
        "MBT_DATABASE_PATH", "data/mbt_evidence.sqlite3"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = EvidenceStore(database_path)
        try:
            yield
        finally:
            app.state.store.close()

    app = FastAPI(title="Configurable FSM Model-Driven Test Backend", version="0.1.0")

    @app.middleware("http")
    async def contract_errors(request: Request, call_next):
        try:
            return await call_next(request)
        except ModelValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    app.router.lifespan_context = lifespan

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/models/validate")
    def validate_model(model: StateMachineModel) -> dict[str, object]:
        transitions = len(model.transitions)
        return {"valid": True, "transitions": transitions, "states": len(model.states)}

    @app.post("/runs", status_code=201)
    def create_run(request: RunRequest) -> dict[str, object]:
        try:
            result = run_model_test(request, app.state.store)
        except ModelValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "category": ErrorCategory.COMPUTATION_FAILURE.value,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            ) from exc
        response = {
            "run_id": result.run_id,
            "status": result.status,
            "failure_category": (
                result.failure_category.value
                if result.failure_category
                else None
            ),
            "failure_step": result.failure_step,
            "seed": result.seed,
            "scheduler": result.scheduler,
            "steps_executed": len(result.steps),
            "reduced_from_step_count": result.reduced_from_step_count,
            "report_url": f"/runs/{result.run_id}",
        }
        if result.replay_script:
            response["replay_url"] = f"/runs/{result.run_id}/replay"
        return response

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[dict[str, object]]:
        limit = min(max(limit, 1), 200)
        return app.state.store.list_runs(limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: int) -> dict[str, object]:
        data = app.state.store.get_run(run_id)
        if data is None:
            raise HTTPException(status_code=404, detail="run not found")
        return data

    @app.get("/runs/{run_id}/replay", response_class=PlainTextResponse)
    def get_replay(run_id: int) -> str:
        script = app.state.store.get_replay_script(run_id)
        if script is None:
            raise HTTPException(status_code=404, detail="replay not found")
        return script

    return app


app = create_app()
