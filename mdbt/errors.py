"""Error contract shared by every module.

The testing backend distinguishes four stable error categories.  Adapters at
module boundaries normalise raw failures (HTTP status codes, exceptions, bad
fixtures) into these categories, so a report never collapses different root
causes into a generic "something failed".

Categories
----------
input_error
    The input model / command is malformed or unknown.  Examples: bad
    transition table, unknown action, unknown resource (SUT HTTP 404/422).
state_conflict
    The input was well formed but illegal in the *current* state.  Examples:
    close twice, submit out of order, cancel a closed resource (SUT HTTP 409).
resource_exhausted
    The backend (or SUT) refused work for a capacity reason: step budget or
    retry budget spent, HTTP 429/503, queue/connection limits.
computation_failure
    An internal invariant broke or the SUT behaved outside the wire contract:
    HTTP 500, malformed response, unexpected exception.
"""
from __future__ import annotations

from enum import Enum


class ErrorCategory(str, Enum):
    INPUT_ERROR = "input_error"
    STATE_CONFLICT = "state_conflict"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMPUTATION_FAILURE = "computation_failure"


class MdbtError(Exception):
    """Base class with a stable, machine-readable ``category``."""

    category: ErrorCategory = ErrorCategory.COMPUTATION_FAILURE

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "message": self.message,
            "detail": self.detail,
        }


class InputError(MdbtError):
    category = ErrorCategory.INPUT_ERROR


class StateConflict(MdbtError):
    category = ErrorCategory.STATE_CONFLICT


class ResourceExhausted(MdbtError):
    category = ErrorCategory.RESOURCE_EXHAUSTED


class ComputationFailure(MdbtError):
    category = ErrorCategory.COMPUTATION_FAILURE


# Wire-status normalisation table used by the executor and both SUTs.
HTTP_STATUS_CATEGORY: dict[int, ErrorCategory] = {
    400: ErrorCategory.INPUT_ERROR,
    404: ErrorCategory.INPUT_ERROR,
    422: ErrorCategory.INPUT_ERROR,
    409: ErrorCategory.STATE_CONFLICT,
    429: ErrorCategory.RESOURCE_EXHAUSTED,
    503: ErrorCategory.RESOURCE_EXHAUSTED,
    500: ErrorCategory.COMPUTATION_FAILURE,
    502: ErrorCategory.COMPUTATION_FAILURE,
}
