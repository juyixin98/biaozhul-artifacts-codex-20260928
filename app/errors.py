from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


class DomainError(Exception):
    code = "DOMAIN_ERROR"
    category = "domain"
    status_code = 400

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidInputError(DomainError):
    code = "INVALID_INPUT"
    category = "invalid_input"
    status_code = 422


class NotFoundError(DomainError):
    code = "NOT_FOUND"
    category = "not_found"
    status_code = 404


class ModelNotFoundError(NotFoundError):
    code = "MODEL_NOT_FOUND"


class BatchNotFoundError(NotFoundError):
    code = "BATCH_NOT_FOUND"


class StateConflictError(DomainError):
    code = "STATE_CONFLICT"
    category = "state_conflict"
    status_code = 409


class ResourceExhaustedError(DomainError):
    code = "RESOURCE_EXHAUSTED"
    category = "resource_exhausted"
    status_code = 413


class ComputationError(DomainError):
    code = "COMPUTATION_FAILED"
    category = "computation_failed"
    status_code = 500


class PersistenceError(DomainError):
    code = "PERSISTENCE_FAILED"
    category = "persistence_failed"
    status_code = 500


def error_body(code: str, category: str, message: str, details: Any, request: Request) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "category": category,
            "message": message,
            "details": details,
            "request_id": request.headers.get("x-request-id", "local"),
        }
    }


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.code, exc.category, exc.message, exc.details, request),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=error_body(
            "INVALID_INPUT",
            "invalid_input",
            "Request fields do not satisfy the Beta-Binomial contract.",
            jsonable_encoder(exc.errors()),
            request,
        ),
    )
