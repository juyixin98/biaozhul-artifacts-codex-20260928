"""Domain and transport error categories.

The categories are part of the API contract. They let callers distinguish a
bad sample from conflicting state, an exhausted local resource, and a numeric
computation failure.
"""

from __future__ import annotations


class DomainError(Exception):
    category = "domain_error"
    status_code = 400

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidInputError(DomainError):
    category = "invalid_input"
    status_code = 422


class StateConflictError(DomainError):
    category = "state_conflict"
    status_code = 409


class NotFoundError(StateConflictError):
    category = "state_conflict"
    status_code = 404


class ResourceExhaustedError(DomainError):
    category = "resource_exhausted"
    status_code = 503


class PayloadTooLargeError(ResourceExhaustedError):
    status_code = 413


class ComputationError(DomainError):
    category = "computation_failure"
    status_code = 500
