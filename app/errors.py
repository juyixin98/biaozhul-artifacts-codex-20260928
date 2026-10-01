"""Domain errors with explicit categories and HTTP status codes."""

from __future__ import annotations

from typing import Any


class CalibrationError(Exception):
    """Base class for expected calibration service failures."""

    category = "CALIBRATION_ERROR"
    code = "calibration_error"
    status_code = 400

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        category: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if category is not None:
            self.category = category
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}


class ContractViolation(CalibrationError):
    category = "VALIDATION_ERROR"
    code = "sample_contract_violation"
    status_code = 422


class ArithmeticConsistencyError(CalibrationError):
    category = "ARITHMETIC_ERROR"
    code = "decomposition_residual_exceeded_tolerance"
    status_code = 500


class ConflictError(CalibrationError):
    category = "PERSISTENCE_ERROR"
    code = "run_id_conflict"
    status_code = 409


class PersistenceError(CalibrationError):
    category = "PERSISTENCE_ERROR"
    code = "sqlite_failure"
    status_code = 500

