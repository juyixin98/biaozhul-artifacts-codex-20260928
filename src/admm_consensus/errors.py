"""Domain error hierarchy shared by the parser, kernel, store, and API."""

from __future__ import annotations


class ErrorCode:
    INVALID_INPUT = "input"
    STATE_CONFLICT = "state_conflict"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMPUTATION_FAILED = "computation"
    NOT_FOUND = "not_found"


class ADMMError(Exception):
    """Base class for expected, categorised application failures."""

    code = ErrorCode.INVALID_INPUT
    http_status = 400

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InputError(ADMMError):
    code = ErrorCode.INVALID_INPUT
    http_status = 422


class StateConflict(ADMMError):
    code = ErrorCode.STATE_CONFLICT
    http_status = 409


class ResourceExhausted(ADMMError):
    code = ErrorCode.RESOURCE_EXHAUSTED
    http_status = 429


class ComputationError(ADMMError):
    code = ErrorCode.COMPUTATION_FAILED
    http_status = 500


class NotFound(ADMMError):
    code = ErrorCode.NOT_FOUND
    http_status = 404
