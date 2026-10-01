"""Typed errors used by the parser, solver, diagnostics and HTTP layer."""

from __future__ import annotations


class MilpError(Exception):
    category = "computation_failed"
    http_status = 500


class InputError(MilpError):
    category = "input_error"
    http_status = 400


class ConfigurationError(MilpError):
    category = "input_error"
    http_status = 400


class StateConflictError(MilpError):
    category = "state_conflict"
    http_status = 409


class ResourceExhaustedError(MilpError):
    category = "resource_exhausted"
    http_status = 409


class ComputationError(MilpError):
    category = "computation_failed"
    http_status = 500
