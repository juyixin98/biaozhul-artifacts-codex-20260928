"""Application error hierarchy.

The same categories are used by the training core, persistence layer, and HTTP
adapter so callers can distinguish bad input from state, resource, or numeric
failures.
"""
from __future__ import annotations


class AppError(Exception):
    """Base class for failures with a stable machine-readable category."""

    category = "application_error"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": self.category,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(AppError):
    """Malformed or semantically invalid request/data."""

    category = "validation_error"


class StateConflictError(AppError):
    """Operation conflicts with existing persisted state."""

    category = "state_conflict"


class ResourceExhaustedError(AppError):
    """A finite local resource needed for progress has been exhausted."""

    category = "resource_exhausted"


class ComputationError(AppError):
    """Numerical computation failed or produced a non-finite result."""

    category = "computation_failed"
