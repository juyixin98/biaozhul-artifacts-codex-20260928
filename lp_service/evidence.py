"""Independent primal, dual, ray and Farkas diagnostics.

These checks deliberately use the original user problem rather than the
transformed simplex tableau, so they can accept or reject a kernel result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Operator
from .parser import StandardLP, VarKind
from .simplex import SimplexResult


@dataclass
class Reconstruction:
    x: np.ndarray
    direction: np.ndarray
    constraint_duals: np.ndarray
    lower_duals: np.ndarray
    upper_duals: np.ndarray


def reconstruct(problem: StandardLP, result: SimplexResult) -> Reconstruction:
    n = problem.n_original
    x = np.zeros(n, dtype=float)
    direction = np.zeros(n, dtype=float)
    alpha = np.zeros(n, dtype=float)
    beta = np.zeros(n, dtype=float)

    for j in range(n):
        lb, ub = problem.original_bounds[j]
        cols = [
            col
            for col, rep in enumerate(problem.variable_reps)
            if rep.original == j
        ]
        plus = [c for c in cols if problem.variable_reps[c].kind == VarKind.FREE_PLUS]
        minus = [c for c in cols if problem.variable_reps[c].kind == VarKind.FREE_MINUS]
        upper_only = [c for c in cols if problem.variable_reps[c].kind == VarKind.UPPER_ONLY]
        normal = [c for c in cols if problem.variable_reps[c].kind == VarKind.NORMAL]

        if plus and minus:
            y = result.x[plus[0]] - result.x[minus[0]]
            d = result.reduced_costs[plus[0]]
            x[j] = y
            if result.standard_direction is not None:
                direction[j] = (
                    result.standard_direction[plus[0]]
                    - result.standard_direction[minus[0]]
                )
            if result.pi is not None:
                _ = d
        elif upper_only:
            col = upper_only[0]
            x[j] = float(ub) - result.x[col]
            if result.standard_direction is not None:
                direction[j] = -result.standard_direction[col]
            if result.pi is not None:
                beta[j] = result.reduced_costs[col]
        elif normal:
            col = normal[0]
            x[j] = float(lb) + result.x[col] if lb is not None else result.x[col]
            if result.standard_direction is not None:
                direction[j] = result.standard_direction[col]
            if result.pi is not None:
                bound_rows = [
                    row
                    for row, source in enumerate(problem.rows)
                    if source.kind == "upper_bound"
                    and problem.variable_reps[col].upper is not None
                    and abs(problem.A[row, col] - 1.0) < 1e-12
                ]
                bound_pi = result.pi[bound_rows[0]] if bound_rows else 0.0
                beta[j] = -bound_pi
                alpha[j] = result.reduced_costs[col]

    f = np.zeros(problem.original_A.shape[0], dtype=float)
    q_or_pi = result.phase1_q if result.phase1_q is not None else result.pi
    if q_or_pi is not None:
        for row, source in enumerate(problem.rows):
            if source.kind == "constraint" and source.original is not None:
                f[source.original] = source.sign * q_or_pi[row]
        if result.phase1_q is not None:
            for j in range(n):
                cols = [
                    col
                    for col, rep in enumerate(problem.variable_reps)
                    if rep.original == j
                ]
                rep = problem.variable_reps[cols[0]]
                if rep.kind == VarKind.UPPER_ONLY:
                    beta[j] = result.reduced_costs[cols[0]]
                elif rep.kind == VarKind.NORMAL:
                    bound_rows = [
                        row
                        for row, source in enumerate(problem.rows)
                        if source.kind == "upper_bound" and abs(problem.A[row, cols[0]] - 1.0) < 1e-12
                    ]
                    bound_pi = result.phase1_q[bound_rows[0]] if bound_rows else 0.0
                    beta[j] = -bound_pi
                    alpha[j] = result.reduced_costs[cols[0]]

    if problem.sense == "max":
        f = -f
        alpha = -alpha
        beta = -beta

    return Reconstruction(x, direction, f, alpha, beta)


def constraint_residuals(problem: StandardLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = problem.original_A @ x
    signed = np.zeros_like(values)
    violations = np.zeros_like(values)
    for i, op in enumerate(problem.original_ops):
        rhs = problem.original_rhs[i]
        if op == "<=":
            signed[i] = rhs - values[i]
            violations[i] = max(0.0, values[i] - rhs)
        elif op == ">=":
            signed[i] = values[i] - rhs
            violations[i] = max(0.0, rhs - values[i])
        else:
            signed[i] = rhs - values[i]
            violations[i] = abs(values[i] - rhs)
    return signed, violations


def bound_violations(problem: StandardLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(problem.n_original, dtype=float)
    upper = np.zeros(problem.n_original, dtype=float)
    for j, (lb, ub) in enumerate(problem.original_bounds):
        if lb is not None:
            lower[j] = max(0.0, lb - x[j])
        if ub is not None:
            upper[j] = max(0.0, x[j] - ub)
    return lower, upper


def original_objective(problem: StandardLP, x: np.ndarray) -> float:
    return float(problem.original_objective_vector() @ x)


def dual_diagnostics(
    problem: StandardLP,
    rec: Reconstruction,
    x: np.ndarray,
    objective_value: float,
    tol: float,
) -> dict[str, float]:
    A = problem.original_A
    c = problem.original_objective_vector()
    stationarity = c - A.T @ rec.constraint_duals - rec.lower_duals + rec.upper_duals
    reduced_violation = float(np.max(np.abs(stationarity))) if stationarity.size else 0.0

    sign_violation = 0.0
    complementarity = 0.0
    for i, op in enumerate(problem.original_ops):
        y = rec.constraint_duals[i]
        if problem.sense == "min":
            if op == "<=":
                sign_violation = max(sign_violation, max(0.0, y))
            elif op == ">=":
                sign_violation = max(sign_violation, max(0.0, -y))
        else:
            if op == "<=":
                sign_violation = max(sign_violation, max(0.0, -y))
            elif op == ">=":
                sign_violation = max(sign_violation, max(0.0, y))
        value = A[i] @ x
        if op == "<=":
            complementarity = max(complementarity, abs(y * (value - problem.original_rhs[i])))
        elif op == ">=":
            complementarity = max(complementarity, abs(y * (value - problem.original_rhs[i])))

    for j, (lb, ub) in enumerate(problem.original_bounds):
        positive_violation = rec.lower_duals if problem.sense == "min" else -rec.lower_duals
        negative_violation = rec.upper_duals if problem.sense == "min" else -rec.upper_duals
        sign_violation = max(sign_violation, max(0.0, -positive_violation[j]))
        sign_violation = max(sign_violation, max(0.0, -negative_violation[j]))
        if lb is not None:
            complementarity = max(complementarity, abs(rec.lower_duals[j] * (lb - x[j])))
        if ub is not None:
            complementarity = max(complementarity, abs(rec.upper_duals[j] * (x[j] - ub)))

    dual_objective = float(rec.constraint_duals @ problem.original_rhs)
    for j, (lb, ub) in enumerate(problem.original_bounds):
        if lb is not None:
            dual_objective += rec.lower_duals[j] * lb
        if ub is not None:
            dual_objective -= rec.upper_duals[j] * ub
    gap = abs(objective_value - dual_objective)

    return {
        "reduced_cost_violation": reduced_violation,
        "dual_sign_violation": float(sign_violation),
        "complementary_slack_violation": float(complementarity),
        "objective_gap": float(gap),
        "dual_tol_ok": float(
            max(reduced_violation, sign_violation, complementarity, gap) <= 100.0 * tol
        ),
    }


def ray_diagnostics(
    problem: StandardLP,
    rec: Reconstruction,
    objective_value: float | None,
    tol: float,
) -> dict[str, float]:
    d = rec.direction
    constraint_step = np.inf
    for i, op in enumerate(problem.original_ops):
        a = problem.original_A[i]
        ad = float(a @ d)
        ax = float(a @ rec.x)
        if op == "<=" and ad > tol:
            constraint_step = min(constraint_step, (problem.original_rhs[i] - ax) / ad)
        elif op == ">=" and ad < -tol:
            constraint_step = min(constraint_step, (ax - problem.original_rhs[i]) / -ad)
        elif op == "=" and abs(ad) > tol:
            constraint_step = 0.0

    bound_step = np.inf
    for j, (lb, ub) in enumerate(problem.original_bounds):
        if lb is not None and d[j] < -tol:
            bound_step = min(bound_step, (rec.x[j] - lb) / -d[j])
        if ub is not None and d[j] > tol:
            bound_step = min(bound_step, (ub - rec.x[j]) / d[j])

    step = min(constraint_step, bound_step)
    improving = -float(problem.original_objective_vector() @ d)
    # A step below zero means x itself was infeasible or the ray is wrong.
    accepted = step >= -100.0 * tol and improving > 100.0 * tol
    return {
        "ray_step": float(max(0.0, step) if np.isfinite(step) else 1.0),
        "ray_improving": float(improving),
        "ray_accepted": float(accepted),
    }


def farkas_diagnostics(
    problem: StandardLP,
    rec: Reconstruction,
    tol: float,
) -> dict[str, float]:
    A = problem.original_A
    # Same stationarity expression as KKT; at a Farkas point it is zero.
    stationarity = -A.T @ rec.constraint_duals - rec.lower_duals + rec.upper_duals
    stationarity_gap = float(np.max(np.abs(stationarity))) if stationarity.size else 0.0
    constant = float(rec.constraint_duals @ problem.original_rhs)
    for j, (lb, ub) in enumerate(problem.original_bounds):
        if lb is not None:
            constant += rec.lower_duals[j] * lb
        if ub is not None:
            constant -= rec.upper_duals[j] * ub

    sign_gap = 0.0
    for i, op in enumerate(problem.original_ops):
        y = rec.constraint_duals[i]
        if op == "<=":
            sign_gap = max(sign_gap, max(0.0, y))
        elif op == ">=":
            sign_gap = max(sign_gap, max(0.0, -y))
    sign_gap = max(sign_gap, float(np.max(-rec.lower_duals)), float(np.max(-rec.upper_duals)))
    accepted = constant > 100.0 * tol and stationarity_gap <= 100.0 * tol and sign_gap <= 100.0 * tol
    return {
        "farkas_constant": constant,
        "farkas_stationarity_gap": stationarity_gap,
        "farkas_sign_gap": sign_gap,
        "farkas_gap": float(max(0.0, 1.0 - float(accepted))),
    }
