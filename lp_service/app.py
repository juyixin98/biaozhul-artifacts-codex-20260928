"""FastAPI entry point."""

import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import SolverConfig
from .logging_setup import configure_logging, redacted_event
from .models import LPRequest
from .solver import solve_lp


def create_app() -> FastAPI:
    app = FastAPI(
        title="Small-Scale Two-Phase Simplex Service",
        version="0.1.0",
    )
    logger = configure_logging()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/solve")
    async def solve(request: Request) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            payload = await request.json()
            model = LPRequest.model_validate(payload)
            result = solve_lp(model, request_id=request_id, config=SolverConfig())
            logger.info(
                redacted_event(
                    request_id,
                    "solve_accepted" if result.response.diagnostic.accepted else "solve_not_verified",
                    {
                        "status": result.response.status,
                        "variables": result.response.solution,
                        "constraints": result.response.residuals,
                        "objective": result.response.objective_value,
                    },
                )
            )
            return JSONResponse(result.response.model_dump(), status_code=200)
        except Exception as exc:
            logger.info(
                redacted_event(
                    request_id,
                    "solve_rejected",
                    {"error_type": type(exc).__name__, "message": str(exc)[:200]},
                )
            )
            return JSONResponse(
                {
                    "status": "invalid_request",
                    "request_id": request_id,
                    "diagnostic": {
                        "accepted": False,
                        "checks": {"request_schema": False},
                        "residuals": {},
                        "message": str(exc),
                    },
                },
                status_code=400,
            )

    return app


app = create_app()
