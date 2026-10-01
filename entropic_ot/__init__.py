"""Entropy-regularized optimal transport package."""

from .kernel import TransportResult, solve_entropic_ot

__all__ = ["TransportResult", "solve_entropic_ot"]

