from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from app.config import Tolerances
from app.core.types import ActiveSetResult
from app.parser import ParsedProblem


@dataclass
class KKTResiduals:
    stationarity_inf: float
    primal_inequality_violation: float
    primal_equality_violation: float
    dual_feasibility_violation: float
    complementarity_max: float
    objective: float
    gradient: list[float]
    equality_multipliers: list[float]
    inequality_multipliers: list[float]
    kkt_active_inequality_indices: list[int]
    solver_active_inequality_indices: list[int]
    active_set_difference: dict[str, list[int]]
    reduced_hessian_condition: float | None
    warnings: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stationarity_inf": self.stationarity_inf,
            "primal_inequality_violation": self.primal_inequality_violation,
            "primal_equality_violation": self.primal_equality_violation,
            "dual_feasibility_violation": self.dual_feasibility_violation,
            "complementarity_max": self.complementarity_max,
            "objective": self.objective,
            "gradient": self.gradient,
            "equality_multipliers": self.equality_multipliers,
            "inequality_multipliers": self.inequality_multipliers,
            "kkt_active_inequality_indices": self.kkt_active_inequality_indices,
            "solver_active_inequality_indices": self.solver_active_inequality_indices,
            "active_set_difference": self.active_set_difference,
            "reduced_hessian_condition": self.reduced_hessian_condition,
            "warnings": self.warnings,
            "failure_reasons": self.failure_reasons,
            "uncertainties": self.uncertainties,
        }


def _condition_number(x: np.ndarray) -> float | None:
    if x.size == 0:
        return None
    values = np.linalg.svd(0.5 * (x + x.T), compute_uv=False)
    if values[-1] <= 0:
        return None
    return float(values[0] / values[-1])


def diagnose(
    problem: ParsedProblem,
    x: np.ndarray,
    active_result: ActiveSetResult,
    tol: Tolerances,
) -> KKTResiduals:
    gradient = problem.H @ x + problem.c
    inequality_residual = problem.G @ x - problem.h
    equality_residual = problem.A @ x - problem.b
    primal_ineq = max(0.0, float(np.max(inequality_residual))) if problem.n_inequalities else 0.0
    primal_eq = float(np.max(np.abs(equality_residual))) if problem.n_equalities else 0.0

    active_mask = inequality_residual >= -tol.active
    active_indices = np.flatnonzero(active_mask).astype(int)
    equality_count = problem.n_equalities
    active_ineq_count = active_indices.size
    M_parts = []
    if equality_count:
        M_parts.extend([problem.A.T, -problem.A.T])
    if active_ineq_count:
        M_parts.append(problem.G[active_indices, :].T)
    M = np.column_stack(M_parts) if M_parts else np.empty((problem.n, 0))

    equality_multipliers = np.zeros(problem.n_equalities)
    inequality_multipliers = np.zeros(problem.n_inequalities)
    stationarity = float(np.linalg.norm(gradient, ord=np.inf))
    if M.shape[1]:
        # Free equality multipliers are represented as positive-minus-negative
        # NNLS components. This keeps inequality duals nonnegative while still
        # handling rank-deficient active sets without an inverse.
        weights, _ = nnls(M, -gradient)
        equality_multipliers = (
            weights[:equality_count] - weights[equality_count : 2 * equality_count]
        )
        inequality_multipliers[active_indices] = weights[2 * equality_count :]
        stationarity = float(np.linalg.norm(
            gradient + problem.A.T @ equality_multipliers + problem.G.T @ inequality_multipliers,
            ord=np.inf,
        ))

    slacks = problem.h - problem.G @ x if problem.n_inequalities else np.asarray([])
    complementarity = (
        float(np.max(np.abs(inequality_multipliers * slacks)))
        if problem.n_inequalities
        else 0.0
    )
    dual_violation = (
        max(0.0, float(-np.min(inequality_multipliers)))
        if problem.n_inequalities
        else 0.0
    )
    objective = float(0.5 * x @ problem.H @ x + problem.c @ x)

    if active_indices.size:
        from app.core.linalg import null_space
        active_equality = np.vstack(
            [problem.A, problem.G[active_indices, :]]
            if problem.n_equalities
            else [problem.G[active_indices, :]]
        )
        Z, _, _ = null_space(active_equality, tol.rank)
        reduced = Z.T @ problem.H @ Z if Z.shape[1] else None
        condition = _condition_number(reduced) if reduced is not None and reduced.size else None
    else:
        condition = _condition_number(problem.H)

    solver_active = set(active_result.active_inequality_indices)
    kkt_active = set(active_indices.tolist())
    failure_reasons: list[str] = []
    uncertainties: list[str] = []

    checks = (
        ("primal inequality", primal_ineq, tol.primal),
        ("primal equality", primal_eq, tol.primal),
        ("stationarity", stationarity, tol.dual),
        ("dual", dual_violation, tol.dual),
        ("complementarity", complementarity, tol.complementarity),
    )
    for label, value, threshold in checks:
        if value > threshold:
            failure_reasons.append(f"{label} residual {value:.3e} exceeds {threshold:.3e}")

    if active_result.status == "uncertain":
        uncertainties.append(active_result.reason)
    if condition is not None and condition > 1.0 / max(tol.rank, np.finfo(float).eps):
        uncertainties.append(f"reduced Hessian condition number {condition:.3e} is near numerical limits")
    return KKTResiduals(
        stationarity_inf=stationarity,
        primal_inequality_violation=primal_ineq,
        primal_equality_violation=primal_eq,
        dual_feasibility_violation=dual_violation,
        complementarity_max=complementarity,
        objective=objective,
        gradient=gradient.tolist(),
        equality_multipliers=equality_multipliers.tolist(),
        inequality_multipliers=inequality_multipliers.tolist(),
        kkt_active_inequality_indices=sorted(kkt_active),
        solver_active_inequality_indices=sorted(solver_active),
        active_set_difference={
            "diagnosed_but_not_in_solver": sorted(kkt_active - solver_active),
            "in_solver_but_not_diagnosed": sorted(solver_active - kkt_active),
        },
        reduced_hessian_condition=condition,
        failure_reasons=failure_reasons,
        uncertainties=uncertainties,
    )
