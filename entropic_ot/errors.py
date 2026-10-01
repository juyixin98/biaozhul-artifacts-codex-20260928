"""Error categories and the contract shared by the kernel and HTTP layer."""

from __future__ import annotations

from typing import Any


class OTError(Exception):
    """Base class for errors with a stable machine-readable category."""

    category = "computation_failed"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.run_id = run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "category": self.category,
                "code": self.__class__.__name__,
                "message": self.message,
                "details": self.details,
                "run_id": self.run_id,
            }
        }


class InputQualityError(OTError):
    category = "input_quality"
    status_code = 422


class StateConflictError(OTError):
    category = "state_conflict"
    status_code = 409


class ResourceExhaustedError(OTError):
    category = "resource_exhausted"
    status_code = 509


class ComputationFailedError(OTError):
    category = "computation_failed"
    status_code = 500

