"""Error taxonomy for the solver service.

Every failure surfaced to clients carries one of these categories so that
input errors, state conflicts, resource exhaustion and computation failures
are distinguishable both in logs and in API responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    INPUT_ERROR = "INPUT_ERROR"
    STATE_CONFLICT = "STATE_CONFLICT"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    COMPUTATION_FAILURE = "COMPUTATION_FAILURE"
    NOT_FOUND = "NOT_FOUND"


class BilpError(Exception):
    """Base class for all service errors."""

    category: ErrorCategory = ErrorCategory.INPUT_ERROR
    http_status: int = 400

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "category": self.category.value,
                "message": self.message,
                "details": self.details,
            }
        }


class InputValidationError(BilpError):
    """Malformed model: bad dimensions, non-finite data, inconsistent bounds."""

    category = ErrorCategory.INPUT_ERROR
    http_status = 400


class StateConflictError(BilpError):
    """Conflicting state, e.g. an idempotency key reused with a different body."""

    category = ErrorCategory.STATE_CONFLICT
    http_status = 409


class ResourceExhaustedError(BilpError):
    """Problem or requested budget exceeds server resource limits."""

    category = ErrorCategory.RESOURCE_EXHAUSTED
    http_status = 413


class ComputationFailureError(BilpError):
    """The numerical kernel failed (LP solver error, internal inconsistency)."""

    category = ErrorCategory.COMPUTATION_FAILURE
    http_status = 502


class RunNotFoundError(BilpError):
    """Unknown run id or node id."""

    category = ErrorCategory.NOT_FOUND
    http_status = 404
