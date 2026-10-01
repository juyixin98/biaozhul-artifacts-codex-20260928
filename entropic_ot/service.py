"""FastAPI entrypoint for the optimal transport solver."""

import logging
import math

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .diagnostics import new_run_id
from .errors import OTError
from .logging_setup import configure_logging
from .schemas import TransportRequest, TransportResponse
from .solver import solve_entropic_ot
from .validation import validate_problem


def create_app() -> FastAPI:
    app = FastAPI(
        title="Entropic Optimal Transport API",
        version="0.1.0",
    )
    logger = configure_logging()

    def json_safe(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        run_id = getattr(request.state, "run_id", None)
        raw_body = (await request.body()).decode("utf-8", errors="replace")
        details = {
            "validation_issues": exc.errors(),
            "raw_body": raw_body,
        }
        logger.warning(
            "request_model_invalid",
            extra={
                "payload": {
                    "run_id": run_id,
                    "error": "input_error",
                    "details": details,
                }
            },
        )
        content = {
            "error": "input_error",
            "message": "Request model validation failed.",
            "run_id": run_id,
            "details": details,
            "diagnostics": None,
        }
        return JSONResponse(status_code=400, content=jsonable_encoder(content))

    @app.exception_handler(OTError)
    async def ot_error_handler(request: Request, exc: OTError) -> JSONResponse:
        run_id = getattr(request.state, "run_id", None) or exc.diagnostics.get("run_id")
        payload = {
            "run_id": run_id,
            "error": exc.category,
            "details": exc.details,
            "diagnostics": exc.diagnostics or None,
        }
        logger.log(
            logging.WARNING if exc.status_code < 500 else logging.ERROR,
            exc.category,
            extra={"payload": payload},
        )
        content = {
            "error": exc.category,
            "message": exc.message,
            "run_id": run_id,
            "details": exc.details,
            "diagnostics": exc.diagnostics or None,
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(json_safe(content)),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        run_id = getattr(request.state, "run_id", None)
        content = {
            "error": "computation_failed",
            "message": "Unexpected computation failure.",
            "run_id": run_id,
            "details": {"exception_type": type(exc).__name__},
            "diagnostics": None,
        }
        logger.exception(
            "unexpected_computation_failed",
            extra={"payload": {"run_id": run_id, "error": "computation_failed"}},
        )
        return JSONResponse(status_code=500, content=jsonable_encoder(content))

    @app.middleware("http")
    async def assign_run_id(request: Request, call_next):
        request.state.run_id = new_run_id()
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/optimal-transport", response_model=TransportResponse)
    async def optimal_transport(
        payload: TransportRequest,
        request: Request,
    ) -> TransportResponse:
        run_id = request.state.run_id
        replay = payload.model_dump()
        logger.info(
            "solve_started",
            extra={
                "payload": {
                    "run_id": run_id,
                    "replay": replay,
                }
            },
        )

        problem = validate_problem(
            payload.source_mass,
            payload.target_mass,
            payload.cost,
            epsilon=payload.epsilon,
            max_iterations=payload.max_iterations,
        )
        result = solve_entropic_ot(
            problem,
            run_id=run_id,
            atol=payload.atol,
            rtol=payload.rtol,
        )
        diagnostics = result.diagnostics.to_dict()
        logger.info(
            "solve_succeeded",
            extra={
                "payload": {
                    "run_id": run_id,
                    "replay": replay,
                    "diagnostics": diagnostics,
                    "transport_cost": result.transport_cost,
                    "regularized_objective": result.regularized_objective,
                }
            },
        )
        return TransportResponse(
            run_id=run_id,
            plan=result.plan.tolist(),
            transport_cost=result.transport_cost,
            regularized_objective=result.regularized_objective,
            entropy_term=result.entropy_term,
            kl_to_uniform=result.kl_to_uniform,
            diagnostics=diagnostics,
        )

    return app


app = create_app()
