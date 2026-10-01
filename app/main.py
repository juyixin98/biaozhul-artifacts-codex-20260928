"""FastAPI entrypoint for the restricted STL backend."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.errors import StlError
from app.schemas import DecomposeRequest, DecomposeResponse, ErrorResponse, VersionResponse
from app.service import DecompositionService
from app.storage import RunRepository


app = FastAPI(
    title="Restricted STL Backend",
    version="1.0.0",
    description="Fixed-period additive STL decomposition with local replay logs.",
)


def get_repository() -> RunRepository:
    return RunRepository(os.environ.get("STL_DATABASE_PATH", "runs/stl_runs.db"))


def get_service(
    repository: RunRepository = Depends(get_repository),
) -> DecompositionService:
    return DecompositionService(repository)


@app.exception_handler(StlError)
async def stl_error_handler(request: Request, exc: StlError) -> JSONResponse:
    payload = ErrorResponse(
        error=exc.error_type,
        category=exc.error_type,
        message=exc.message,
        details=exc.details,
        run_id=getattr(request.state, "run_id", None),
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "INPUT_ERROR",
            "category": "INPUT_ERROR",
            "message": "request does not satisfy the sample contract",
            "details": {"validation_errors": exc.errors()},
            "run_id": None,
        },
    )


@app.exception_handler(json.JSONDecodeError)
async def json_decode_error_handler(
    request: Request, exc: json.JSONDecodeError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "INPUT_ERROR",
            "category": "INPUT_ERROR",
            "message": "request body is not valid JSON",
            "details": {"json_position": exc.pos},
            "run_id": None,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version", response_model=VersionResponse)
def version(service: DecompositionService = Depends(get_service)) -> dict[str, Any]:
    return service.version()


@app.post("/decompose", response_model=DecomposeResponse, status_code=201)
def decompose(
    request: DecomposeRequest,
    http_request: Request,
    service: DecompositionService = Depends(get_service),
) -> dict[str, Any]:
    http_request.state.run_id = request.run_id
    run_id, response, _ = service.decompose(request.run_id, request.values)
    http_request.state.run_id = run_id
    return response


@app.get("/runs/{run_id}")
def get_run(
    run_id: str, repository: RunRepository = Depends(get_repository)
) -> dict[str, Any]:
    run = repository.get_run(run_id)
    if run is None:
        from app.errors import InputError

        raise InputError("run not found", {"run_id": run_id})
    return run

