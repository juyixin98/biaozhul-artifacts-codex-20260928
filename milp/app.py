"""FastAPI service entrypoint."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import get_settings
from .errors import MilpError
from .model import parse_problem
from .response import build_response
from .schemas import SolveRequest
from .solver import solve

app = FastAPI(
    title="Bounded Integer Linear Programming API",
    version="1.0.0",
    description="Branch-and-bound solver with SciPy HiGHS LP relaxations and replayable JSONL journals.",
)


@app.exception_handler(MilpError)
async def milp_error_handler(_: Request, exc: MilpError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": {"category": exc.category, "message": str(exc)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": {"category": "input_error", "message": "request validation failed", "details": exc.errors()}},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/solve")
def api_solve(request: SolveRequest):
    settings = get_settings()
    result = solve(request.problem, request.options, settings=settings)
    parsed = parse_problem(request.problem)
    return build_response(result, parsed.names, settings.max_events_in_response)
