"""FastAPI entry point for bounded nonlinear least squares."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .diagnostics import RunLogger
from .expression import ExpressionValidationError, IllegalFunctionError
from .schemas import HealthResponse, SolveRequest, SolveResponse
from .solver import SolveOptions, solve_nonlinear_least_squares

settings = get_settings()
app = FastAPI(
    title="Bounded Nonlinear Least Squares",
    version=__version__,
    description="Scaled trust-region nonlinear least squares with local synthetic data.",
)


@app.exception_handler(ExpressionValidationError)
async def expression_validation_handler(_: Request, exc: ExpressionValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "status": "INVALID_REQUEST",
            "reason": str(exc),
        },
    )


@app.exception_handler(IllegalFunctionError)
async def illegal_function_handler(_: Request, exc: IllegalFunctionError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "status": "ILLEGAL_FUNCTION_VALUE",
            "reason": str(exc),
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service=settings.service_name, status="ok", version=__version__)


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> dict:
    names = list(request.parameters.keys())
    initial = {name: spec.initial for name, spec in request.parameters.items()}
    lower = {name: spec.lower for name, spec in request.parameters.items()}
    upper = {name: spec.upper for name, spec in request.parameters.items()}
    scales = {
        name: spec.scale
        for name, spec in request.parameters.items()
        if spec.scale is not None
    }
    logger = RunLogger(path=settings.log_path, run_id=request.run_id)
    result = solve_nonlinear_least_squares(
        expression=request.expression,
        parameter_names=names,
        x=np.asarray(request.x, dtype=float),
        y=np.asarray(request.y, dtype=float),
        initial=initial,
        lower=lower,
        upper=upper,
        scales=scales,
        options=SolveOptions(
            max_iterations=request.options.max_iterations,
            tolerance=request.options.tolerance,
            initial_trust_radius=request.options.initial_trust_radius,
            max_trust_radius=request.options.max_trust_radius,
            min_trust_radius=request.options.min_trust_radius,
            derivative_check=request.options.derivative_check,
            derivative_tolerance=request.options.derivative_tolerance,
            rank_tolerance=request.options.rank_tolerance,
        ),
        logger=logger,
        run_id=request.run_id,
    )
    payload = result.to_dict()
    payload["run_id"] = result.diagnostics.run_id
    return payload
