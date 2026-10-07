"""Error contract shared by every module.

Five failure classes are distinguished by ``code``:

    INPUT_ERROR        caller data is malformed or semantically invalid (400)
    NOT_FOUND          a referenced word/corpus/run does not exist (404)
    STATE_CONFLICT     the operation contradicts stored state (409)
    RESOURCE_EXHAUSTED a declared resource limit was exceeded (413)
    COMPUTATION_FAILED numerical failure such as non-finite values (422)

Each error carries a stable ``code``, a human readable ``message`` and a
JSON-serialisable ``details`` mapping so test logs can record the exact
failure category without parsing exception text.
"""

from __future__ import annotations

from typing import Any


class SGNSError(Exception):
    """Base class for all expected backend failures."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class InputError(SGNSError):
    code = "INPUT_ERROR"
    http_status = 400


class NotFoundError(SGNSError):
    code = "NOT_FOUND"
    http_status = 404


class StateConflictError(SGNSError):
    code = "STATE_CONFLICT"
    http_status = 409


class ResourceExhaustedError(SGNSError):
    code = "RESOURCE_EXHAUSTED"
    http_status = 413


class ComputationFailureError(SGNSError):
    code = "COMPUTATION_FAILED"
    http_status = 422
