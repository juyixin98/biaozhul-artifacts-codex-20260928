"""Entropically regularized optimal transport package."""

from .solver import TransportResult, solve_entropic_ot
from .validation import ValidatedProblem, validate_problem

__all__ = [
    "TransportResult",
    "ValidatedProblem",
    "solve_entropic_ot",
    "validate_problem",
]

