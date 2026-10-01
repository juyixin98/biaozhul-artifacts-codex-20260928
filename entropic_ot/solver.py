"""Numerical kernel for finite discrete entropically regularized OT.

The optimization problem is

    minimize  <P, C> + epsilon * sum_ij P_ij (log(P_ij) - 1)
    subject to P 1 = a, P^T 1 = b, P >= 0.

Updates retain dual potentials and use log-sum-exp.  A row/column cost gauge
keeps the required edges finite even when individual costs are much larger
than epsilon.  It changes potentials by additive constants but leaves P and
the original transport cost unchanged.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .diagnostics import Diagnostics, EventCollector, IterationEvent
from .errors import ComputationFailedError, ResourceExhaustedError
from .validation import ValidatedProblem


@dataclass(frozen=True, slots=True)
class TransportResult:
    plan: np.ndarray
    transport_cost: float
    regularized_objective: float
    entropy_term: float
    kl_to_uniform: float
    diagnostics: Diagnostics


def _cost_gauge(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a row/column-shifted cost with a finite zero in each line."""

    row_min = np.min(cost, axis=1, keepdims=True)
    shifted = cost - row_min
    column_min = np.min(shifted, axis=0, keepdims=True)
    shifted = shifted - column_min
    if not np.all(np.isfinite(shifted)):
        raise ComputationFailedError(
            "Cost gauge produced non-finite values.",
            details={"shifted_min": float(np.nanmin(shifted))},
        )
    gauge = row_min[:, 0, None] + column_min[0, None, :]
    return shifted, gauge


def _line_errors(
    log_marginals: np.ndarray,
    targets: np.ndarray,
    *,
    atol: float,
    rtol: float,
    total_mass: float,
) -> tuple[float, float, np.ndarray]:
    """Absolute and stable log-space marginal errors."""

    tolerance = atol * max(total_mass, 1.0) + rtol * np.maximum(targets, 0.0)
    floor = atol * max(total_mass, 1.0)
    raw_error = np.zeros_like(targets)

    positive = targets > 0.0
    if np.any(positive):
        raw_error[positive] = (
            np.abs(np.expm1(log_marginals[positive] - np.log(targets[positive])))
            * targets[positive]
        )
    zero_targets = ~positive
    if np.any(zero_targets):
        zero_log_marginals = log_marginals[zero_targets]
        raw_error[zero_targets] = np.where(
            np.isneginf(zero_log_marginals),
            0.0,
            np.exp(np.minimum(zero_log_marginals, 0.0)),
        )

    relative_scale = np.maximum(targets, floor)
    ratios = raw_error / relative_scale
    normalized = raw_error / np.maximum(tolerance, np.finfo(float).tiny)
    return float(np.max(raw_error)), float(np.max(ratios)), normalized


def _entropy_and_objective(
    plan: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
) -> tuple[float, float, float]:
    positive = plan > 0.0
    entropy = float(np.sum(plan[positive] * (np.log(plan[positive]) - 1.0)))
    transport_cost = float(np.sum(plan * cost))
    return (
        transport_cost,
        transport_cost + epsilon * entropy,
        entropy,
    )


def _failed_diagnostics(
    *,
    run_id: str,
    reason: str,
    iterations: int,
    problem: ValidatedProblem,
    active_shape: tuple[int, int],
    row_error: float,
    column_error: float,
    combined: float,
    atol: float,
    events: list[IterationEvent],
) -> Diagnostics:
    return Diagnostics(
        run_id=run_id,
        status="failed",
        stopped_reason=reason,
        feasible=False,
        converged=False,
        iterations=iterations,
        max_iterations=problem.max_iterations,
        active_shape=active_shape,
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        combined_error_ratio=combined,
        row_tolerance=atol,
        column_tolerance=atol,
        events=events,
    )


def _computation_failure(
    message: str,
    *,
    diagnostics: Diagnostics,
    details: dict[str, Any] | None = None,
) -> ComputationFailedError:
    return ComputationFailedError(
        message,
        details=details or {},
        diagnostics=diagnostics.to_dict(),
    )


def solve_entropic_ot(
    problem: ValidatedProblem,
    *,
    run_id: str,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> TransportResult:
    """Solve a validated entropic OT problem with explicit feasibility evidence."""

    if not isinstance(atol, (int, float)) or isinstance(atol, bool) or atol < 0.0:
        raise ComputationFailedError(
            "atol must be a non-negative number.",
            details={"actual": atol},
        )
    if not isinstance(rtol, (int, float)) or isinstance(rtol, bool) or rtol < 0.0:
        raise ComputationFailedError(
            "rtol must be a non-negative number.",
            details={"actual": rtol},
        )
    if not np.isfinite(atol) or not np.isfinite(rtol) or (atol == 0.0 and rtol == 0.0):
        raise ComputationFailedError(
            "At least one finite positive convergence tolerance is required.",
            details={"atol": float(atol), "rtol": float(rtol)},
        )

    full_shape = (problem.source_mass.size, problem.target_mass.size)
    active_rows = np.flatnonzero(problem.active_rows)
    active_columns = np.flatnonzero(problem.active_columns)
    active_shape = (active_rows.size, active_columns.size)

    if problem.total_mass == 0.0:
        diagnostics = Diagnostics(
            run_id=run_id,
            status="converged",
            stopped_reason="all_masses_zero",
            feasible=True,
            converged=True,
            iterations=0,
            max_iterations=problem.max_iterations,
            active_shape=active_shape,
            row_marginal_error=0.0,
            column_marginal_error=0.0,
            combined_error_ratio=0.0,
            row_tolerance=atol,
            column_tolerance=atol,
        )
        zero_plan = np.zeros(full_shape, dtype=np.float64)
        return TransportResult(zero_plan, 0.0, 0.0, 0.0, 0.0, diagnostics)

    a = problem.source_mass[active_rows]
    b = problem.target_mass[active_columns]
    active_cost = problem.active_cost
    shifted_cost, _ = _cost_gauge(active_cost)
    scaled_cost = shifted_cost / problem.epsilon
    if not np.all(np.isfinite(scaled_cost)):
        failure_diagnostics = _failed_diagnostics(
            run_id=run_id,
            reason="cost_epsilon_scale_overflow",
            iterations=0,
            problem=problem,
            active_shape=active_shape,
            row_error=float("inf"),
            column_error=float("inf"),
            combined=float("inf"),
            atol=atol,
            events=[],
        )
        raise _computation_failure(
            "Cost divided by epsilon overflowed; the scale is outside float64 support.",
            diagnostics=failure_diagnostics,
            details={
                "epsilon": problem.epsilon,
                "shifted_cost_max": float(np.max(shifted_cost)),
            },
        )

    log_a = np.log(a)
    log_b = np.log(b)
    f = np.zeros(active_shape[0], dtype=np.float64)
    g = np.zeros(active_shape[1], dtype=np.float64)
    collector = EventCollector()

    row_error = float("inf")
    column_error = float("inf")
    iterations = 0

    for iterations in range(1, problem.max_iterations + 1):
        # Row dual update: log(a_i) = f_i + logsumexp_j(g_j - S_ij/e).
        with np.errstate(over="ignore", invalid="ignore"):
            f = log_a - logsumexp(-scaled_cost + g[None, :], axis=1)
            f -= np.mean(f)

            # Column dual update: log(b_j) = g_j + logsumexp_i(f_i - S_ij/e).
            g = log_b - logsumexp(-scaled_cost + f[:, None], axis=0)

            log_plan = f[:, None] + g[None, :] - scaled_cost

        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
            failure_diagnostics = _failed_diagnostics(
                run_id=run_id,
                reason="non_finite_dual_potentials",
                iterations=iterations,
                problem=problem,
                active_shape=active_shape,
                row_error=row_error,
                column_error=column_error,
                combined=float("inf"),
                atol=atol,
                events=collector.events,
            )
            raise _computation_failure(
                "Sinkhorn dual potentials became non-finite.",
                diagnostics=failure_diagnostics,
                details={"iteration": iterations},
            )

        log_row_mass = logsumexp(log_plan, axis=1)
        log_column_mass = logsumexp(log_plan, axis=0)
        if not np.all(np.isfinite(log_row_mass)) or not np.all(np.isfinite(log_column_mass)):
            failure_diagnostics = _failed_diagnostics(
                run_id=run_id,
                reason="non_finite_log_marginals",
                iterations=iterations,
                problem=problem,
                active_shape=active_shape,
                row_error=row_error,
                column_error=column_error,
                combined=float("inf"),
                atol=atol,
                events=collector.events,
            )
            raise _computation_failure(
                "Log-domain marginals became non-finite.",
                diagnostics=failure_diagnostics,
                details={"iteration": iterations},
            )

        row_error, _, row_normalized = _line_errors(
            log_row_mass,
            a,
            atol=atol,
            rtol=rtol,
            total_mass=problem.total_mass,
        )
        column_error, _, column_normalized = _line_errors(
            log_column_mass,
            b,
            atol=atol,
            rtol=rtol,
            total_mass=problem.total_mass,
        )
        combined = float(max(np.max(row_normalized), np.max(column_normalized)))
        collector.add(
            IterationEvent(
                iteration=iterations,
                row_marginal_error=row_error,
                column_marginal_error=column_error,
                combined_error_ratio=combined,
                finite_log_plan=bool(np.all(np.isfinite(log_plan))),
            )
        )

        if combined <= 1.0:
            active_plan = np.exp(log_plan)
            break
    else:
        diagnostics = Diagnostics(
            run_id=run_id,
            status="failed",
            stopped_reason="iteration_budget_exhausted",
            feasible=False,
            converged=False,
            iterations=problem.max_iterations,
            max_iterations=problem.max_iterations,
            active_shape=active_shape,
            row_marginal_error=row_error,
            column_marginal_error=column_error,
            combined_error_ratio=combined,
            row_tolerance=atol,
            column_tolerance=atol,
            events=collector.events,
        )
        raise ResourceExhaustedError(
            "Iteration budget exhausted before both marginals converged.",
            details={
                "iterations": problem.max_iterations,
                "row_marginal_error": row_error,
                "column_marginal_error": column_error,
            },
            diagnostics=diagnostics.to_dict(),
        )

    active_plan = np.exp(log_plan)
    if np.any(active_plan < 0.0) or not np.all(np.isfinite(active_plan)):
        failure_diagnostics = _failed_diagnostics(
            run_id=run_id,
            reason="non_finite_materialized_plan",
            iterations=iterations,
            problem=problem,
            active_shape=active_shape,
            row_error=row_error,
            column_error=column_error,
            combined=combined,
            atol=atol,
            events=collector.events,
        )
        raise _computation_failure(
            "Materialized transport plan contains negative, NaN, or infinite values.",
            diagnostics=failure_diagnostics,
            details={"iteration": iterations},
        )

    actual_rows = np.sum(active_plan, axis=1)
    actual_columns = np.sum(active_plan, axis=0)
    row_tolerance = atol * max(problem.total_mass, 1.0) + rtol * a
    column_tolerance = atol * max(problem.total_mass, 1.0) + rtol * b
    feasible = (
        np.all(np.abs(actual_rows - a) <= row_tolerance)
        and np.all(np.abs(actual_columns - b) <= column_tolerance)
    )
    if not feasible:
        diagnostics = _failed_diagnostics(
            run_id=run_id,
            reason="materialized_marginals_failed",
            iterations=iterations,
            problem=problem,
            active_shape=active_shape,
            row_error=float(np.max(np.abs(actual_rows - a))),
            column_error=float(np.max(np.abs(actual_columns - b))),
            combined=combined,
            atol=atol,
            events=collector.events,
        )
        raise _computation_failure(
            "Log-domain convergence did not survive materialization of the transport plan.",
            diagnostics=diagnostics,
        )

    plan = np.zeros(full_shape, dtype=np.float64)
    plan[np.ix_(active_rows, active_columns)] = active_plan
    # Enforce zero-mass row/column contracts exactly rather than leaving roundoff.
    plan[~problem.active_rows, :] = 0.0
    plan[:, ~problem.active_columns] = 0.0

    transport_cost, regularized_objective, entropy = _entropy_and_objective(
        plan,
        problem.cost,
        problem.epsilon,
    )
    positive_plan = plan[plan > 0.0]
    kl_to_uniform = float(
        np.sum(
            positive_plan
            * np.log(positive_plan * plan.size / problem.total_mass)
        )
    )

    diagnostics = Diagnostics(
        run_id=run_id,
        status="converged",
        stopped_reason="marginal_tolerance_satisfied",
        feasible=True,
        converged=True,
        iterations=iterations,
        max_iterations=problem.max_iterations,
        active_shape=active_shape,
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        combined_error_ratio=combined,
        row_tolerance=atol,
        column_tolerance=atol,
        events=collector.events,
    )
    return TransportResult(
        plan=plan,
        transport_cost=transport_cost,
        regularized_objective=regularized_objective,
        entropy_term=entropy,
        kl_to_uniform=max(0.0, kl_to_uniform),
        diagnostics=diagnostics,
    )
