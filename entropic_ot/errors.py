"""Shared error categories and exception contracts."""

from typing import Any, Literal


ErrorCategory = Literal[
    "input_error",
    "state_conflict",
    "resource_exhausted",
    "computation_failed",
]


class OTError(Exception):
    """Base class for errors that have a stable machine-readable category."""

    category: ErrorCategory = "computation_failed"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.diagnostics = diagnostics or {}


class InputValidationError(OTError):
    category = "input_error"
    status_code = 400


class StateConflictError(OTError):
    category = "state_conflict"
    status_code = 409


class ResourceExhaustedError(OTError):
    category = "resource_exhausted"
    status_code = 503


class ComputationFailedError(OTError):
    category = "computation_failed"
    status_code = 500

