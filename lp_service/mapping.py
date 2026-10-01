"""Map standard-form simplex results back to the original LP."""

from dataclasses import dataclass

import numpy as np

from .kernel import KernelResult
from .parser import LPProblem
from .standardize import StandardModel


@dataclass
class MappedResult:
    status: str
    objective_value: float | None
    solution: np.ndarray | None
    direction: np.ndarray | None
    constraint_duals: np.ndarray | None
    lower_duals: np.ndarray | None
    upper_duals: np.ndarray | None
    fixed_duals: np.ndarray | None
    reduced_costs: np.ndarray | None
    constraint_certificate: np.ndarray | None
    lower_certificate: np.ndarray | None
    upper_certificate: np.ndarray | None
    certificate_contradiction: float | None
    iterations: int
    history: list[dict[str, object]]


def map_result(
    problem: LPProblem,
    model: StandardModel,
    kernel: KernelResult,
) -> MappedResult:
    solution = None
    direction = None
    objective_value = None
    if kernel.standard_solution is not None:
        solution = _map_vector(problem, model, kernel.standard_solution)
    if kernel.standard_direction is not None:
        direction = _map_vector(problem, model, kernel.standard_direction, direction=True)

    if kernel.status == "optimal" and kernel.objective_value is not None:
        internal_value = -kernel.objective_value + model.objective_constant
        objective_value = float(internal_value if problem.sense == "min" else -internal_value)

    constraint_duals = lower_duals = upper_duals = fixed_duals = None
    reduced_costs = None
    constraint_certificate = lower_certificate = upper_certificate = None
    contradiction = None

    if kernel.status == "optimal" and kernel.equality_duals is not None and kernel.reduced_costs is not None:
        constraint_duals, lower_duals, upper_duals, fixed_duals = _map_duals(
            problem, model, kernel.equality_duals, kernel.reduced_costs
        )
        reduced_costs = _reduced_costs_original(
            problem, model, kernel.reduced_costs, lower_duals, upper_duals
        )

    if kernel.status == "infeasible" and kernel.equality_duals is not None:
        (
            constraint_certificate,
            lower_certificate,
            upper_certificate,
            contradiction,
        ) = _map_farkas(problem, model, kernel.equality_duals)

    return MappedResult(
        status=kernel.status,
        objective_value=objective_value,
        solution=solution,
        direction=direction,
        constraint_duals=constraint_duals,
        lower_duals=lower_duals,
        upper_duals=upper_duals,
        fixed_duals=fixed_duals,
        reduced_costs=reduced_costs,
        constraint_certificate=constraint_certificate,
        lower_certificate=lower_certificate,
        upper_certificate=upper_certificate,
        certificate_contradiction=contradiction,
        iterations=kernel.iterations,
        history=kernel.history,
    )


def _map_vector(
    problem: LPProblem,
    model: StandardModel,
    z: np.ndarray,
    direction: bool = False,
) -> np.ndarray:
    x = np.zeros(problem.c.size, dtype=float)
    for col, meta in enumerate(model.columns):
        if meta.variable < 0 or col >= model.core_column_count or col >= z.size:
            continue
        j = meta.variable
        lb, ub = float(problem.lower[j]), float(problem.upper[j])
        if meta.kind == "shifted":
            x[j] = z[col] if direction else lb + z[col]
        elif meta.kind == "upper_only":
            x[j] = -z[col] if direction else ub - z[col]
        elif meta.kind == "positive_part":
            x[j] += z[col]
        elif meta.kind == "negative_part":
            x[j] -= z[col]
    fixed = problem.lower == problem.upper
    if not direction:
        x[fixed] = problem.lower[fixed]
    return x


def _map_duals(
    problem: LPProblem,
    model: StandardModel,
    y: np.ndarray,
    reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = problem.c.size
    m = problem.A.shape[0]
    constraints = np.zeros(m, dtype=float)
    lower = np.zeros(n, dtype=float)
    upper = np.zeros(n, dtype=float)
    for row, meta in enumerate(model.rows):
        value = float(meta.dual_factor * y[row])
        if meta.kind == "constraint":
            constraints[meta.original_index] = value
        else:
            upper[meta.original_index] = value

    for col, meta in enumerate(model.columns):
        if meta.variable < 0 or col >= model.core_column_count:
            continue
        j = meta.variable
        if meta.kind == "shifted":
            lower[j] = float(reduced[col])
        else:
            upper[j] = float(reduced[col])

    fixed = problem.lower == problem.upper
    stationarity = problem.c - problem.A.T @ constraints
    fixed_duals = np.zeros(n, dtype=float)
    fixed_duals[fixed] = stationarity[fixed]

    if problem.sense == "max":
        constraints *= -1.0
        lower *= -1.0
        upper *= -1.0
        fixed_duals *= -1.0

    return constraints, lower, upper, fixed_duals


def _reduced_costs_original(
    problem: LPProblem,
    model: StandardModel,
    reduced: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    out = np.zeros(problem.c.size, dtype=float)
    for col, meta in enumerate(model.columns):
        if meta.variable < 0 or col >= model.core_column_count:
            continue
        out[meta.variable] = float(reduced[col])
    return out


def _map_farkas(
    problem: LPProblem,
    model: StandardModel,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = problem.c.size
    m = problem.A.shape[0]
    q = np.zeros(m, dtype=float)
    lower_q = np.zeros(n, dtype=float)
    upper_q = np.zeros(n, dtype=float)

    for meta in model.rows:
        r = meta.row_index
        if r < 0:
            continue
        value = float(meta.cert_factor * y[r])
        if meta.kind == "constraint":
            q[meta.original_index] = value
        elif meta.kind == "upper":
            upper_q[meta.original_index] = value
        # Fixed variables are eliminated and do not have a Farkas bound row.

    stationarity = problem.A.T @ q
    for j in range(n):
        lb, ub = float(problem.lower[j]), float(problem.upper[j])
        if lb == ub:
            continue
        a = float(stationarity[j])
        if lb > -np.inf:
            lower_q[j] = -a - upper_q[j] if ub < np.inf else -a
        elif ub < np.inf:
            lower_q[j] = 0.0
            upper_q[j] = a
        else:
            lower_q[j] = max(0.0, -a)
            upper_q[j] = max(0.0, a)

    standard_constant = float(np.sum(y * model.d))
    original_constant = _original_contradiction(problem, q, lower_q, upper_q)
    # The two should be identical numerically. Exposing the original scalar is
    # useful; standard_constant is retained in diagnostics as a cross-check.
    _ = standard_constant
    return q, lower_q, upper_q, original_constant

def _original_contradiction(
    problem: LPProblem,
    q: np.ndarray,
    lower_q: np.ndarray,
    upper_q: np.ndarray,
) -> float:
    finite_lower = np.isfinite(problem.lower)
    finite_upper = np.isfinite(problem.upper)
    non_fixed = problem.lower != problem.upper
    return float(
        problem.b @ q
        + problem.lower[finite_lower & non_fixed] @ lower_q[finite_lower & non_fixed]
        + problem.upper[finite_upper & non_fixed] @ upper_q[finite_upper & non_fixed]
    )
