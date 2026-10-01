"""Domain error types and stable error codes.

Every failure surfaced by the service carries one of these codes so clients
can react programmatically instead of parsing free-text messages. Unexpected
exceptions are mapped to INTERNAL_ERROR by the service layer and are never
silently converted into successful responses.
"""
from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    EMPTY_GROUP = "EMPTY_GROUP"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    INVALID_REQUEST = "INVALID_REQUEST"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    RUN_CANCELLED = "RUN_CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DomainError(Exception):
    """An expected, client-visible failure with a stable code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict | None = None,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.http_status = http_status
