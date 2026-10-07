"""Shared error categories used by the analyzer, runner, and report API."""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    """Stable values persisted in SQLite and returned by HTTP APIs."""

    INPUT_ERROR = "input_error"
    STATE_CONFLICT = "state_conflict"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMPUTATION_FAILURE = "computation_failure"
    RESOURCE_NOT_FOUND = "resource_not_found"


class ModelValidationError(ValueError):
    """The supplied state-machine model is internally inconsistent."""

    category = ErrorCategory.INPUT_ERROR


class StateConflictError(RuntimeError):
    """The system-under-test rejected an operation in its current state."""

    category = ErrorCategory.STATE_CONFLICT


class ResourceExhaustedError(RuntimeError):
    """A local, synthetic resource limit was reached."""

    category = ErrorCategory.RESOURCE_EXHAUSTED


class ComputationFailureError(RuntimeError):
    """Transport, serialization, or a local deterministic check failed."""

    category = ErrorCategory.COMPUTATION_FAILURE


class ResourceNotFoundError(LookupError):
    """A local synthetic resource id does not exist."""

    category = ErrorCategory.RESOURCE_NOT_FOUND
