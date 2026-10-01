"""Typed error contract shared by the kernel, repository and API."""

from __future__ import annotations


class StlError(Exception):
    """Base class for errors that have stable machine-readable categories."""

    code = "calculation_failed"
    status_code = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InputContractError(StlError):
    code = "input_contract_violation"
    status_code = 400


class StateConflictError(StlError):
    code = "state_conflict"
    status_code = 409


class ResourceExhaustedError(StlError):
    code = "resource_exhausted"
    status_code = 507


class CalculationError(StlError):
    code = "calculation_failed"
    status_code = 422
