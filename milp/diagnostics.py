"""Independent feasibility and numerical diagnostics for candidate solutions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import ParsedProblem


@dataclass(frozen=True)
class CandidateCheck:
    feasible: bool
    failure_category: str | None
    max_integer_residual: float
    max_bound_violation: float
    max_inequality_violation: float
    max_equality_residual: float
    internal_objective: float | None
    original_objective: float | None
    message: str | None = None


def _max_residual(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def verify_candidate(
    problem: ParsedProblem,
    values: np.ndarray,
    *,
    integer_tolerance: float,
    constraint_tolerance: float,
) -> CandidateCheck:
    values = np.asarray(values, dtype=float)
    if values.shape != (problem.n,):
        raise ValueError("candidate length does not match problem")
    if not np.all(np.isfinite(values)):
        return CandidateCheck(False, "non_finite", math.inf, math.inf, math.inf, math.inf, None, None, "candidate is not finite")

    integer_values = values[problem.integer]
    integer_residual = _max_residual(integer_values - np.rint(integer_values)) if integer_values.size else 0.0

    lower_violation = np.maximum(problem.original_lower - values, 0.0)
    upper_violation = np.maximum(values - problem.original_upper, 0.0)
    bound_violation = _max_residual(np.concatenate((lower_violation, upper_violation)))

    if problem.a_ub.shape[0]:
        inequality_violation = float(np.max(problem.a_ub @ values - problem.b_ub))
    else:
        inequality_violation = 0.0
    inequality_violation = max(0.0, inequality_violation)

    if problem.a_eq.shape[0]:
        equality_residual = _max_residual(problem.a_eq @ values - problem.b_eq)
    else:
        equality_residual = 0.0

    internal_obj = float(problem.objective @ values)
    original_obj = -internal_obj if problem.sense == "maximize" else internal_obj

    if integer_residual > integer_tolerance:
        return CandidateCheck(
            False,
            "integer_tolerance",
            integer_residual,
            bound_violation,
            inequality_violation,
            equality_residual,
            internal_obj,
            original_obj,
            f"integer residual {integer_residual:.12g} exceeds tolerance {integer_tolerance:.12g}",
        )
    if bound_violation > constraint_tolerance:
        return CandidateCheck(
            False,
            "bound_violation",
            integer_residual,
            bound_violation,
            inequality_violation,
            equality_residual,
            internal_obj,
            original_obj,
            f"bound violation {bound_violation:.12g} exceeds tolerance {constraint_tolerance:.12g}",
        )
    if inequality_violation > constraint_tolerance:
        return CandidateCheck(
            False,
            "inequality_violation",
            integer_residual,
            bound_violation,
            inequality_violation,
            equality_residual,
            internal_obj,
            original_obj,
            f"inequality violation {inequality_violation:.12g} exceeds tolerance {constraint_tolerance:.12g}",
        )
    if equality_residual > constraint_tolerance:
        return CandidateCheck(
            False,
            "equality_violation",
            integer_residual,
            bound_violation,
            inequality_violation,
            equality_residual,
            internal_obj,
            original_obj,
            f"equality residual {equality_residual:.12g} exceeds tolerance {constraint_tolerance:.12g}",
        )

    return CandidateCheck(
        True,
        None,
        integer_residual,
        bound_violation,
        inequality_violation,
        equality_residual,
        internal_obj,
        original_obj,
        None,
    )


def rounded_integer_candidate(values: np.ndarray, integer: np.ndarray) -> np.ndarray:
    candidate = np.array(values, dtype=float, copy=True)
    candidate[integer] = np.rint(candidate[integer])
    return candidate
