"""Exceptions with stable, machine-readable error codes."""


class PermutationError(Exception):
    """Base class for expected service errors."""

    error_code = "PERMUTATION_ERROR"
    status_code = 500
    category = "computation"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidInputError(PermutationError):
    error_code = "INVALID_INPUT"
    status_code = 400
    category = "invalid_input"


class ExactLimitExceededError(PermutationError):
    error_code = "EXACT_LIMIT_EXCEEDED"
    status_code = 400
    category = "invalid_request"


class ComputationError(PermutationError):
    error_code = "COMPUTATION_FAILED"
    status_code = 500
    category = "computation"


class CancellationError(PermutationError):
    error_code = "CANCELED"
    status_code = 409
    category = "cancellation"


class RunNotFoundError(PermutationError):
    error_code = "RUN_NOT_FOUND"
    status_code = 404
    category = "not_found"


class RunConflictError(PermutationError):
    error_code = "RUN_CONFLICT"
    status_code = 409
    category = "conflict"

