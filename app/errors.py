"""Domain and API error types."""

from __future__ import annotations

from typing import Any


class NsgaError(Exception):
    """Base error carrying a stable machine-readable code and context."""

    status_code = 422
    code = "invalid_request"
    stage = "validation"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        stage: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        if stage is not None:
            self.stage = stage
        self.details = details or {}


class ValidationFailure(NsgaError):
    status_code = 422
    code = "validation_failed"


class NotFound(NsgaError):
    status_code = 404
    code = "not_found"


class Conflict(NsgaError):
    status_code = 409
    code = "conflict"
