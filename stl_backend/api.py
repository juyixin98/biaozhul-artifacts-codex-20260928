"""FastAPI interface for the constrained STL backend."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .config import DEFAULT_CONFIG, DEFAULT_DATABASE_PATH
from .errors import StlError
from .repository import RunRepository
from .schemas import DecompositionRequest
from .service import decompose_values


def create_app(database_path: str = DEFAULT_DATABASE_PATH) -> FastAPI:
    app = FastAPI(
        title="Constrained STL Decomposition API",
        version="1.0.0",
        openapi_tags=[
            {"name": "decomposition", "description": "Run fixed STL decomposition"},
            {"name": "runs", "description": "Inspect persisted run diagnostics"},
        ],
    )
    app.state.database_path = database_path

    def repository() -> RunRepository:
        repo = RunRepository(app.state.database_path)
        try:
            yield repo
        finally:
            repo.close()

    @app.exception_handler(StlError)
    async def stl_error_handler(_: Request, exc: StlError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "category": exc.code.rsplit("_", 1)[0]
                    if exc.code != "state_conflict"
                    else "state",
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "input_contract_violation",
                    "category": "input",
                    "message": "request does not satisfy the decomposition schema",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(sqlite3.Error)
    async def sqlite_error_handler(_: Request, exc: sqlite3.Error) -> JSONResponse:
        return JSONResponse(
            status_code=507,
            content={
                "error": {
                    "code": "resource_exhausted",
                    "category": "resource",
                    "message": "SQLite run storage failed",
                    "details": {"database_error": str(exc)},
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "kernel_version": DEFAULT_CONFIG.kernel_version}

    @app.get("/api/v1/config", tags=["decomposition"])
    async def get_config() -> dict[str, Any]:
        return {
            "schema_version": DEFAULT_CONFIG.schema_version,
            "kernel_version": DEFAULT_CONFIG.kernel_version,
            "config": DEFAULT_CONFIG.as_public_dict(),
        }

    @app.post("/api/v1/decompositions", tags=["decomposition"])
    async def create_decomposition(request: DecompositionRequest) -> dict[str, Any]:
        return decompose_values(
            request.values,
            run_id=request.run_id,
            config=DEFAULT_CONFIG,
            database_path=app.state.database_path,
        )

    @app.get("/api/v1/runs/{run_id}", tags=["runs"])
    async def get_run(
        run_id: str, repo: RunRepository = Depends(repository)
    ) -> JSONResponse:
        row = repo.get_run(run_id)
        if row is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "run_not_found",
                        "category": "state",
                        "message": "unknown run_id",
                        "details": {"run_id": run_id},
                    }
                },
            )

        result = json.loads(row["result_json"]) if row["result_json"] else None
        return JSONResponse(
            content={
                "run_id": row["run_id"],
                "run_number": row["id"],
                "status": row["status"],
                "observations": row["observations"],
                "request_fingerprint": row["request_fingerprint"],
                "kernel_version": row["kernel_version"],
                "schema_version": row["schema_version"],
                "error": {
                    "category": row["error_category"],
                    "code": row["error_code"],
                    "message": row["error_message"],
                }
                if row["status"] == "failed"
                else None,
                "result": result,
            }
        )

    @app.get("/api/v1/runs/{run_id}/events", tags=["runs"])
    async def get_run_events(
        run_id: str, repo: RunRepository = Depends(repository)
    ) -> JSONResponse:
        row = repo.get_run(run_id)
        if row is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "run_not_found",
                        "category": "state",
                        "message": "unknown run_id",
                        "details": {"run_id": run_id},
                    }
                },
            )
        events = [
            {
                "id": event["id"],
                "run_id": event["run_id"],
                "run_number": event["run_number"],
                "iteration": event["iteration"],
                "stage": event["stage"],
                "judgment": event["judgment"],
                "state": json.loads(event["state_json"]),
                "created_at": event["created_at"],
            }
            for event in repo.list_events(run_id)
        ]
        return JSONResponse(
            content={
                "run_id": run_id,
                "run_number": row["id"],
                "event_count": len(events),
                "events": events,
            }
        )

    return app


app = create_app()
