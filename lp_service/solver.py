"""High-level solve orchestration and evidence acceptance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import evidence
from .models import SolveRequest, SolveResponse
from .parser import StandardLP, standardize
from .simplex import KernelStatus, run_two_phase


@dataclass
class SolveOutcome:
    request_id: str
    status: str
    message: str
    objective_value: float | None
    solution: np.ndarray | None
    direction: np.ndarray | None
    constraint_duals: np.ndarray | None
    lower_duals: np.ndarray | None
    upper_duals: np.ndarray | None
    constraint_signed_residuals: np.ndarray | None
    constraint_violations: np.ndarray | None
    lower_violations: np.ndarray | None
    upper_violations: np.ndarray | None
    metrics: dict[str, float]
    iterations: int
    pivots: int
    artificial_remaining: int
    redundant_rows: list[int]
    pivot_log: list[dict[str, float | int | str]]
    problem: StandardLP | None = None
    rejected: list[str] = field(default_factory=list)


def _tol(request: SolveRequest, problem: StandardLP) -> float:
    scale = max(
        1.0,
        float(np.max(np.abs(problem.original_A))) if problem.original_A.size else 1.0,
        float(np.max(np.abs(problem.original_rhs))) if problem.original_rhs.size else 1.0,
        float(np.max(np.abs(problem.original_c))),
    )
    return request.tolerance or 1e-9 * scale


def solve_core(request: SolveRequest) -> SolveOutcome:
    problem = standardize(request)
    tol = _tol(request, problem)
    kernel = run_two_phase(problem, request.max_iterations, request.tolerance)
    rec = evidence.reconstruct(problem, kernel)
    _, constraint_violations = evidence.constraint_residuals(problem, rec.x)
    lower_violations, upper_violations = evidence.bound_violations(problem, rec.x)
    residual = float(np.max(constraint_violations)) if constraint_violations.size else 0.0
    bound_violation = float(
        max(
            np.max(lower_violations) if lower_violations.size else 0.0,
            np.max(upper_violations) if upper_violations.size else 0.0,
        )
    )
    request_id = request.request_id or "local-solve"
    objective_value = evidence.original_objective(problem, rec.x)

    metrics: dict[str, float] = {
        "primal_residual": residual,
        "bound_violation": bound_violation,
    }

    def outcome(
        status: str,
        message: str,
        *,
        include_solution: bool = True,
        include_duals: bool = False,
        direction: np.ndarray | None = None,
        rejected: list[str] | None = None,
    ) -> SolveOutcome:
        return SolveOutcome(
            request_id=request_id,
            status=status,
            message=message,
            objective_value=objective_value if include_solution else None,
            solution=rec.x if include_solution else None,
            direction=direction,
            constraint_duals=rec.constraint_duals if include_duals else None,
            lower_duals=rec.lower_duals if include_duals else None,
            upper_duals=rec.upper_duals if include_duals else None,
            constraint_signed_residuals=None,
            constraint_violations=constraint_violations,
            lower_violations=lower_violations,
            upper_violations=upper_violations,
            metrics=dict(metrics),
            iterations=kernel.iterations,
            pivots=kernel.pivots,
            artificial_remaining=kernel.artificial_remaining,
            redundant_rows=kernel.redundant_rows,
            pivot_log=kernel.pivot_log,
            problem=problem,
            rejected=rejected or [],
        )

    if kernel.status == KernelStatus.INFEASIBLE:
        farkas = evidence.farkas_diagnostics(problem, rec, tol)
        metrics.update(farkas)
        if farkas["farkas_gap"] > 0.0:
            return outcome(
                "numerical_uncertainty",
                "kernel reported infeasible but the Farkas certificate failed",
                include_solution=False,
                rejected=["invalid_farkas_certificate"],
            )
        result = outcome("infeasible", "infeasible, certified by Farkas multipliers", include_solution=False)
        result.constraint_duals = rec.constraint_duals
        result.lower_duals = rec.lower_duals
        result.upper_duals = rec.upper_duals
        return result

    if kernel.status == KernelStatus.NUMERICAL:
        return outcome(
            "numerical_uncertainty",
            kernel.message,
            include_solution=residual <= 100.0 * tol and bound_violation <= 100.0 * tol,
            rejected=["kernel_numerical_uncertainty"],
        )

    primal_ok = residual <= 100.0 * tol and bound_violation <= 100.0 * tol
    if not primal_ok:
        return outcome(
            "numerical_uncertainty",
            "kernel returned a point that fails independent primal verification",
            include_solution=True,
            rejected=["primal_verification_failed"],
        )

    if kernel.status == KernelStatus.UNBOUNDED:
        ray = evidence.ray_diagnostics(problem, rec, objective_value, tol)
        metrics.update(ray)
        if ray["ray_accepted"] < 0.5:
            return outcome(
                "numerical_uncertainty",
                "kernel reported unbounded but the improving ray failed verification",
                direction=rec.direction,
                rejected=["invalid_unbounded_ray"],
            )
        return outcome(
            "unbounded",
            "unbounded in the improving direction",
            direction=rec.direction,
        )

    if kernel.status != KernelStatus.OPTIMAL:
        return outcome("numerical_uncertainty", "unknown kernel state", rejected=["unknown_kernel_status"])

    duals = evidence.dual_diagnostics(problem, rec, rec.x, objective_value, tol)
    metrics.update(duals)
    if duals["dual_tol_ok"] < 0.5:
        return outcome(
            "numerical_uncertainty",
            "optimal primal point failed independent KKT/dual verification",
            include_duals=True,
            rejected=["dual_verification_failed"],
        )

    return outcome("optimal", kernel.message, include_duals=True)


def _map_vector(names: list[str], values: np.ndarray | None) -> dict[str, float] | None:
    if values is None:
        return None
    return {name: float(value) for name, value in zip(names, values)}


def solve_lp(request: SolveRequest) -> SolveResponse:
    outcome = solve_core(request)
    problem = outcome.problem
    residuals = None
    bound_diagnostics = None
    if problem is not None and outcome.constraint_violations is not None:
        signed, _ = evidence.constraint_residuals(
            problem, outcome.solution if outcome.solution is not None else np.zeros(problem.n_original)
        )
        residuals = [
            {
                "name": name,
                "op": op,
                "rhs": float(rhs),
                "residual": float(signed[i]),
                "violated": bool(outcome.constraint_violations[i] > 1e-8),
            }
            for i, (name, op, rhs) in enumerate(
                zip(problem.constraint_names, problem.original_ops, problem.original_rhs)
            )
        ]
        lower = outcome.lower_violations
        upper = outcome.upper_violations
        x = outcome.solution if outcome.solution is not None else np.zeros(problem.n_original)
        bound_diagnostics = [
            {
                "variable": name,
                "lower": bounds[0],
                "upper": bounds[1],
                "value": float(x[j]),
                "lower_violation": float(lower[j] if lower is not None else 0.0),
                "upper_violation": float(upper[j] if upper is not None else 0.0),
            }
            for j, (name, bounds) in enumerate(zip(problem.original_names, problem.original_bounds))
        ]

    evidence_model = {
        "primal_residual": outcome.metrics.get("primal_residual", 0.0),
        "bound_violation": outcome.metrics.get("bound_violation", 0.0),
        "objective_gap": outcome.metrics.get("objective_gap"),
        "reduced_cost_violation": outcome.metrics.get("reduced_cost_violation"),
        "dual_sign_violation": outcome.metrics.get("dual_sign_violation"),
        "complementary_slack_violation": outcome.metrics.get("complementary_slack_violation"),
        "ray_step": outcome.metrics.get("ray_step"),
        "farkas_gap": outcome.metrics.get("farkas_gap"),
    }
    return SolveResponse(
        request_id=outcome.request_id,
        status=outcome.status,  # type: ignore[arg-type]
        message=(
            outcome.message
            if not outcome.rejected
            else outcome.message + "; rejected evidence=" + ",".join(outcome.rejected)
        ),
        objective_value=outcome.objective_value,
        solution=_map_vector(problem.original_names if problem else [], outcome.solution),
        unbounded_direction=_map_vector(
            problem.original_names if problem else [], outcome.direction
        ),
        constraint_duals=_map_vector(
            problem.constraint_names if problem else [], outcome.constraint_duals
        ),
        lower_duals=_map_vector(
            problem.original_names if problem else [], outcome.lower_duals
        ),
        upper_duals=_map_vector(
            problem.original_names if problem else [], outcome.upper_duals
        ),
        residuals=residuals,  # type: ignore[arg-type]
        bound_diagnostics=bound_diagnostics,  # type: ignore[arg-type]
        evidence=evidence_model,  # type: ignore[arg-type]
        iterations=outcome.iterations,
        pivots=outcome.pivots,
        artificial_remaining=outcome.artificial_remaining,
        redundant_rows=outcome.redundant_rows,
        pivot_log=outcome.pivot_log,
    )
