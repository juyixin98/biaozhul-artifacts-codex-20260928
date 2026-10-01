from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.errors import (
    DomainError,
    domain_error_handler,
    error_body,
    validation_error_handler,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local Beta-Binomial Backend",
        version="1.0.0",
        description="Conjugate Beta posterior updates, batch revision, and prediction.",
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"local-{uuid.uuid4().hex}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    app.add_exception_handler(DomainError, domain_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_body(
                "COMPUTATION_FAILED",
                "computation_failed",
                "An unexpected local computation failure occurred.",
                {"exception": exc.__class__.__name__, "message": str(exc)},
                request,
            ),
        )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)
    return app


app = create_app()
