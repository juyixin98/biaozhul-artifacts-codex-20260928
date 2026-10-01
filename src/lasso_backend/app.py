"""FastAPI entry point."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import get_settings
from .errors import LassoRequestError
from .models import LassoPathRequest, LassoPathResponse
from .preprocessing import PreprocessingError
from .service import run_lasso
from .solver import SolverError

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="Lasso Path Backend",
    version="0.1.0",
    summary="Coordinate-descent Lasso path with warm starts and KKT diagnostics",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}


@app.post(
    "/api/v1/lasso/path",
    response_model=LassoPathResponse,
    tags=["lasso"],
)
def fit_lasso_path_http(
    request: LassoPathRequest, http_request: Request
) -> LassoPathResponse:
    return run_lasso(request, settings, http_request)


def _rejected_response(
    http_request: Request,
    exc: Exception,
    category: str,
    status_code: int = 422,
) -> JSONResponse:
    request_id = getattr(http_request.state, "request_id", None)
    return JSONResponse(
        status_code=status_code,
        content={
            "accepted": False,
            "decision": "rejected",
            "error_category": category,
            "message": str(exc),
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    http_request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "accepted": False,
            "decision": "rejected",
            "error_category": "invalid_request_contract",
            "message": "request payload does not satisfy the API contract",
            "details": jsonable_encoder(exc.errors()),
            "request_id": str(uuid.uuid4()),
        },
    )


@app.exception_handler(LassoRequestError)
async def request_error_handler(
    http_request: Request, exc: LassoRequestError
) -> JSONResponse:
    return _rejected_response(http_request, exc, "invalid_request")


@app.exception_handler(PreprocessingError)
async def preprocessing_error_handler(
    http_request: Request, exc: PreprocessingError
) -> JSONResponse:
    return _rejected_response(http_request, exc, "invalid_preprocessing")


@app.exception_handler(SolverError)
async def solver_error_handler(
    http_request: Request, exc: SolverError
) -> JSONResponse:
    return _rejected_response(http_request, exc, "invalid_solver_configuration")
