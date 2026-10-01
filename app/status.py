"""Explicit solver and protocol statuses."""

from enum import Enum


class Status(str, Enum):
    CONVERGED = "converged"
    NONCONVERGENT = "nonconvergent"
    UNIDENTIFIABLE = "unidentifiable"
    ILLEGAL_FUNCTION_VALUE = "illegal_function_value"
    DERIVATIVE_CHECK_FAILED = "derivative_check_failed"
    INVALID_EXPRESSION = "invalid_expression"
    INVALID_REQUEST = "invalid_request"
    INTERNAL_ERROR = "internal_error"


FAILING_STATUSES = {
    Status.NONCONVERGENT,
    Status.UNIDENTIFIABLE,
    Status.ILLEGAL_FUNCTION_VALUE,
    Status.DERIVATIVE_CHECK_FAILED,
    Status.INVALID_EXPRESSION,
    Status.INVALID_REQUEST,
    Status.INTERNAL_ERROR,
}


class SolverError(Exception):
    """An error carrying an explicit machine-readable solver status."""

    def __init__(
        self,
        status: Status,
        message: str,
        *,
        details: dict | None = None,
        http_status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details or {}
        self.http_status_code = http_status_code
