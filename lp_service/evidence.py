"""Independent primal, dual, ray and Farkas evidence verification."""

from dataclasses import dataclass

import numpy as np

from .config import SolverConfig
from .mapping import MappedResult
from .parser import LPProblem


@dataclass
class EvidenceReport:
    accepted: bool
    checks: dict[str, bool]
    residuals: dict[str, float | list[float]]
    message: str


def verify_result(
    problem: LPProblem,
    result: MappedResult,
    config: SolverConfig,
) -> EvidenceReport:
    if result.status == "optimal":
        return _verify_optimal(problem, result, config)
    if result.status == "infeasible":
        return _verify_infeasible(problem, result, config)
    if result.status == "unbounded":
        return _verify_unbounded(problem, result, config)
    return EvidenceReport(
        accepted=False,
        checks={"numerical_uncertain_kept_separate": True},
        residuals={},
        message="The kernel could neither accept nor reject the status within tolerance.",
    )


def primal_residuals(problem: LPProblem, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ax = problem.A @ x
    constraint = np.zeros(problem.A.shape[0])
    for i, sense in enumerate(problem.constraint_senses):
        if sense == "le":
            constraint[i] = ax[i] - problem.b[i]
        elif sense == "ge":
            constraint[i] = problem.b[i] - ax[i]
        else:
            constraint[i] = ax[i] - problem.b[i]

    bound = np.zeros(problem.c.size)
    finite_lower = np.isfinite(problem.lower)
    finite_upper = np.isfinite(problem.upper)
    bound[finite_lower] = np.maximum(0.0, problem.lower[finite_lower] - x[finite_lower])
    bound[finite_upper] = np.maximum(bound[finite_upper], x[finite_upper] - problem.upper[finite_upper])
    return constraint, bound


def objective_value(problem: LPProblem, x: np.ndarray) -> float:
    return float(problem.c @ x)


def _verify_optimal(problem: LPProblem, result: MappedResult, config: SolverConfig) -> EvidenceReport:
    assert result.solution is not None
    assert result.objective_value is not None
    residual, bound_violation = primal_residuals(problem, result.solution)
    computed_objective = objective_value(problem, result.solution)
    objective_error = abs(computed_objective - result.objective_value)

    checks = {
        "constraints_feasible": bool(np.max(residual, initial=0.0) <= config.evidence_tol),
        "bounds_feasible": bool(np.max(bound_violation, initial=0.0) <= config.evidence_tol),
        "objective_matches_point": objective_error <= config.evidence_tol * max(1.0, abs(computed_objective)),
    }

    if result.constraint_duals is not None and result.lower_duals is not None and result.upper_duals is not None:
        stationarity = (
            problem.c
            - problem.A.T @ result.constraint_duals
            - result.lower_duals
            - result.upper_duals
        )
        fixed = problem.lower == problem.upper
        stationarity[fixed] = 0.0
        sign_ok = _dual_signs_ok(problem, result)
        dual_objective = _dual_objective(problem, result)
        gap = abs(dual_objective - result.objective_value)
        checks.update(
            {
                "dual_stationarity": bool(np.max(np.abs(stationarity), initial=0.0) <= config.evidence_tol),
                "dual_signs": sign_ok,
                "weak_duality_gap_zero": gap <= config.evidence_tol * max(1.0, abs(result.objective_value)),
            }
        )

    accepted = all(checks.values())
    return EvidenceReport(
        accepted=accepted,
        checks=checks,
        residuals={
            "constraint_residuals": residual.tolist(),
            "bound_violations": bound_violation.tolist(),
            "objective_error": objective_error,
        },
        message="Optimal status accepted from primal and dual evidence."
        if accepted
        else "At least one independent optimal-status check failed.",
    )


def _dual_signs_ok(problem: LPProblem, result: MappedResult) -> bool:
    assert result.constraint_duals is not None
    assert result.lower_duals is not None
    assert result.upper_duals is not None
    tol = 1e-7
    # Convention:
    # min: <= y <= 0, >= y >= 0; lower lambda >= 0, upper gamma <= 0.
    # max: inequality and bound multiplier signs are reversed.
    factor = -1.0 if problem.sense == "max" else 1.0
    for i, sense in enumerate(problem.constraint_senses):
        y = factor * result.constraint_duals[i]
        if sense == "le" and y > tol:
            return False
        if sense == "ge" and y < -tol:
            return False
    if np.any(factor * result.lower_duals < -tol):
        return False
    if np.any(factor * result.upper_duals > tol):
        return False
    return True


def _dual_objective(problem: LPProblem, result: MappedResult) -> float:
    assert result.constraint_duals is not None
    assert result.lower_duals is not None
    assert result.upper_duals is not None
    # Equalities are naturally free. Fixed variables were substituted, so
    # their contribution is present in the shifted RHS term.
    finite_lower = np.isfinite(problem.lower)
    finite_upper = np.isfinite(problem.upper)
    non_fixed = problem.lower != problem.upper
    value = float(problem.b @ result.constraint_duals)
    fixed = problem.lower == problem.upper
    value += float(problem.lower[fixed] @ result.fixed_duals[fixed])
    value += float(problem.lower[finite_lower & non_fixed] @ result.lower_duals[finite_lower & non_fixed])
    value += float(problem.upper[finite_upper & non_fixed] @ result.upper_duals[finite_upper & non_fixed])
    return value


def _verify_unbounded(problem: LPProblem, result: MappedResult, config: SolverConfig) -> EvidenceReport:
    assert result.solution is not None
    assert result.direction is not None
    residual, bound_violation = primal_residuals(problem, result.solution)
    directional_residual, directional_bound = _direction_feasibility(problem, result.direction)
    slope = float(problem.c @ result.direction)
    expected_slope_positive = problem.sense == "min"
    improves = slope < -config.evidence_tol if expected_slope_positive else slope > config.evidence_tol
    nonzero = float(np.linalg.norm(result.direction)) > config.evidence_tol

    checks = {
        "witness_feasible": bool(np.max(np.abs(residual), initial=0.0) <= config.evidence_tol)
        and bool(np.max(residual, initial=0.0) <= config.evidence_tol)
        and bool(np.max(bound_violation, initial=0.0) <= config.evidence_tol),
        "ray_preserves_equalities": bool(
            np.max(np.abs(directional_residual), initial=0.0) <= config.evidence_tol
        ),
        "ray_preserves_bounds": bool(np.max(directional_bound, initial=0.0) <= config.evidence_tol),
        "ray_nonzero": nonzero,
        "objective_improves": improves,
    }
    accepted = all(checks.values())
    return EvidenceReport(
        accepted=accepted,
        checks=checks,
        residuals={
            "witness_constraint_residuals": residual.tolist(),
            "witness_bound_violations": bound_violation.tolist(),
            "directional_constraint_residuals": directional_residual.tolist(),
            "directional_bound_violations": directional_bound.tolist(),
            "objective_slope": slope,
        },
        message="Unbounded ray independently verified."
        if accepted
        else "The proposed unbounded ray failed an independent check.",
    )


def _direction_feasibility(problem: LPProblem, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ad = problem.A @ d
    constraint = np.zeros(problem.A.shape[0])
    for i, sense in enumerate(problem.constraint_senses):
        if sense == "eq":
            constraint[i] = ad[i]
        elif sense == "le":
            constraint[i] = max(0.0, ad[i])
        else:
            constraint[i] = max(0.0, -ad[i])

    bound = np.zeros(problem.c.size)
    for j in range(problem.c.size):
        lb, ub = problem.lower[j], problem.upper[j]
        if np.isfinite(lb) and np.isfinite(ub):
            bound[j] = 0.0
        elif np.isfinite(lb) and d[j] < 0.0:
            bound[j] = -d[j]
        elif np.isfinite(ub) and d[j] > 0.0:
            bound[j] = d[j]
    return constraint, bound


def _verify_infeasible(problem: LPProblem, result: MappedResult, config: SolverConfig) -> EvidenceReport:
    assert result.constraint_certificate is not None
    assert result.lower_certificate is not None
    assert result.upper_certificate is not None
    assert result.certificate_contradiction is not None

    q = result.constraint_certificate
    lower = result.lower_certificate
    upper = result.upper_certificate
    stationarity = problem.A.T @ q + lower + upper
    fixed = problem.lower == problem.upper
    stationarity[fixed] = 0.0

    sign_ok = _certificate_signs_ok(problem, q, lower, upper, config.evidence_tol)
    contradiction = result.certificate_contradiction
    checks = {
        "certificate_nonnegative_by_sense": sign_ok,
        "farkas_stationarity_zero": bool(
            np.max(np.abs(stationarity), initial=0.0) <= config.evidence_tol
        ),
        # Multipliers are normalized to the all-">=" Farkas convention:
        # q^T A + lambda + gamma = 0 and q^T b + ... > 0.
        "farkas_contradiction_positive": contradiction > config.evidence_tol,
    }
    accepted = all(checks.values())
    return EvidenceReport(
        accepted=accepted,
        checks=checks,
        residuals={
            "stationarity": stationarity.tolist(),
            "constraint_certificate": q.tolist(),
            "lower_certificate": lower.tolist(),
            "upper_certificate": upper.tolist(),
            "contradiction": contradiction,
        },
        message="Infeasibility accepted from a Farkas certificate."
        if accepted
        else "Proposed infeasibility certificate failed independent verification.",
    )


def _certificate_signs_ok(
    problem: LPProblem,
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tol: float,
) -> bool:
    for i, sense in enumerate(problem.constraint_senses):
        if sense != "eq" and q[i] < -tol:
            return False
    non_fixed = problem.lower != problem.upper
    if np.any(lower[non_fixed] < -tol) or np.any(upper[non_fixed] < -tol):
        return False
    return True
