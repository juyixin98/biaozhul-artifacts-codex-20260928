"""Error categories shared by the API, storage, parser and solver."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for errors with a stable machine-readable contract."""

    category = "internal_error"
    code = "internal_error"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}

    def to_response(self) -> dict[str, Any]:
        return {
            "error": {
                "category": self.category,
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class InputContractError(AppError):
    category = "input_error"
    code = "invalid_input"
    status_code = 400


class ModelParseError(InputContractError):
    code = "invalid_quadratic_model"


class StateConflictError(AppError):
    category = "state_conflict"
    code = "invalid_state_transition"
    status_code = 409


class NotFoundError(AppError):
    category = "input_error"
    code = "not_found"
    status_code = 404


class ResourceLimitError(AppError):
    category = "resource_exhausted"
    code = "resource_limit_exceeded"
    status_code = 507


class ComputationContractError(AppError):
    category = "computation_failure"
    code = "computation_failed"
    status_code = 422


class KernelComputationError(AppError):
    category = "computation_failure"
    code = "kernel_computation_failed"
    status_code = 500
