"""Domain and HTTP error types shared by the kernel, service, and API."""

from __future__ import annotations

from typing import Any


class StlError(Exception):
    """Base class for errors with a stable, testable category."""

    error_type = "COMPUTATION_FAILED"
    status_code = 422

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InputError(StlError):
    error_type = "INPUT_ERROR"
    status_code = 400


class ResourceExhaustedError(StlError):
    error_type = "RESOURCE_EXHAUSTED"
    status_code = 413


class ComputationError(StlError):
    error_type = "COMPUTATION_FAILED"
    status_code = 422


class StateConflictError(StlError):
    error_type = "STATE_CONFLICT"
    status_code = 409


class StorageError(StlError):
    error_type = "STORAGE_FAILED"
    status_code = 500

