"""Numerical core for entropy-regularized discrete optimal transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .diagnostics import RunRecorder
from .errors import (
    ComputationFailedError,
    InputQualityError,
    OTError,
    ResourceExhaustedError,
)


@dataclass(frozen=True)
class TransportResult:
    plan: np.ndarray
    original_cost: float
    regularized_cost: float
    entropy_penalty: float
    epsilon: float
    iterations: int
    max_iterations: int
    converged: bool
    feasible: bool
    stop_reason: str
    row_error: float
    column_error: float
    row_marginals: np.ndarray
    column_marginals: np.ndarray
    diagnostics: dict[str, Any]


def solve_entropic_ot(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    cost: np.ndarray | list[list[float]],
    epsilon: float,
    *,
    max_iterations: int = 100_000,
    tolerance: float = 1e-10,
    recorder: RunRecorder | None = None,
    include_logged_inputs: bool = False,
) -> TransportResult:
    """Solve a balanced entropy-regularized discrete transport problem."""

    try:
        a_vec, b_vec, c_mat, epsilon, max_iterations, tolerance = _validate_inputs(
            a, b, cost, epsilon, max_iterations, tolerance
        )
        if recorder is not None:
            recorder.start(
                {
                    "parameters": {
                        "epsilon": epsilon,
                        "max_iterations": max_iterations,
                        "tolerance": tolerance,
                    },
                    "input_shape": {
                        "rows": int(a_vec.size),
                        "columns": int(b_vec.size),
                        "cost_shape": list(c_mat.shape),
                    },
                    "inputs": _jsonable_arrays(a_vec, b_vec, c_mat)
                    if include_logged_inputs
                    else None,
                }
            )

        n, m = a_vec.size, b_vec.size
        total_mass = float(a_vec.sum())
        if total_mass == 0.0:
            result = _zero_mass_result((n, m), epsilon, max_iterations, tolerance)
            if recorder is not None:
                recorder.finish(_result_to_log(result))
            return result

        active_rows = np.flatnonzero(a_vec > 0.0)
        active_cols = np.flatnonzero(b_vec > 0.0)
        aa = a_vec[active_rows]
        bb = b_vec[active_cols]
        cc = c_mat[np.ix_(active_rows, active_cols)]

        log_a = np.log(aa)
        log_b = np.log(bb)
        log_kernel = -cc / epsilon
        f = np.zeros(active_rows.size, dtype=np.float64)
        g = np.zeros(active_cols.size, dtype=np.float64)

        iterations = 0
        log_row_error = np.inf
        log_col_error = np.inf
        converged = False

        for iterations in range(1, max_iterations + 1):
            f = log_a - logsumexp(log_kernel + g[None, :], axis=1)
            g = log_b - logsumexp(log_kernel + f[:, None], axis=0)

            log_plan = log_kernel + f[:, None] + g[None, :]
            if not np.all(np.isfinite(log_plan)):
                raise ComputationFailedError(
                    "A non-finite log transport weight was produced.",
                    details={"iteration": iterations},
                    run_id=recorder.run_id if recorder is not None else None,
                )

            log_row_error = _relative_max_error(
                aa, np.exp(logsumexp(log_plan, axis=1)), tolerance, total_mass
            )
            log_col_error = _relative_max_error(
                bb, np.exp(logsumexp(log_plan, axis=0)), tolerance, total_mass
            )

            if recorder is not None:
                recorder.iteration(
                    iterations,
                    {
                        "log_row_error": float(log_row_error),
                        "log_column_error": float(log_col_error),
                        "f_min": float(f.min()),
                        "f_max": float(f.max()),
                        "g_min": float(g.min()),
                        "g_max": float(g.max()),
                        "log_plan_min": float(log_plan.min()),
                        "log_plan_max": float(log_plan.max()),
                    },
                    total_iterations=max_iterations,
                )

            if log_row_error <= 1.0 and log_col_error <= 1.0:
                converged = True
                break

        active_plan = _reconstruct_plan(
            log_kernel, f, g, aa, bb, repair_columns=converged
        )
        plan = np.zeros((n, m), dtype=np.float64)
        plan[np.ix_(active_rows, active_cols)] = active_plan

        row_marginals = plan.sum(axis=1)
        col_marginals = plan.sum(axis=0)
        row_error = _max_absolute_error(a_vec, row_marginals)
        col_error = _max_absolute_error(b_vec, col_marginals)
        feasible = bool(
            _within_tolerance(a_vec, row_marginals, tolerance, total_mass)
            and _within_tolerance(b_vec, col_marginals, tolerance, total_mass)
        )
        if converged and not feasible:
            converged = False

        stop_reason = (
            "converged"
            if converged
            else "iteration_budget_exhausted"
            if iterations >= max_iterations
            else "not_converged"
        )

        original_cost = float(np.sum(plan * c_mat))
        entropy_penalty = _kl_penalty(plan, a_vec, b_vec, total_mass)
        regularized_cost = original_cost + epsilon * entropy_penalty
        if not all(
            np.isfinite(x) for x in (original_cost, entropy_penalty, regularized_cost)
        ):
            raise ComputationFailedError(
                "The objective contains a non-finite value.",
                details={
                    "original_cost": original_cost,
                    "entropy_penalty": entropy_penalty,
                    "regularized_cost": regularized_cost,
                },
                run_id=recorder.run_id if recorder is not None else None,
            )

        result = TransportResult(
            plan=plan,
            original_cost=original_cost,
            regularized_cost=float(regularized_cost),
            entropy_penalty=float(entropy_penalty),
            epsilon=epsilon,
            iterations=int(iterations),
            max_iterations=int(max_iterations),
            converged=bool(converged),
            feasible=feasible,
            stop_reason=stop_reason,
            row_error=float(row_error),
            column_error=float(col_error),
            row_marginals=row_marginals,
            column_marginals=col_marginals,
            diagnostics={
                "tolerance": tolerance,
                "total_mass": total_mass,
                "active_rows": active_rows.tolist(),
                "active_columns": active_cols.tolist(),
                "log_row_error_at_stop": float(log_row_error),
                "log_column_error_at_stop": float(log_col_error),
                "plan_mass": float(plan.sum()),
                "feasibility_reason": _feasibility_reason(
                    a_vec, b_vec, row_marginals, col_marginals, tolerance, total_mass
                ),
            },
        )
        if recorder is not None:
            recorder.finish(_result_to_log(result))
        return result

    except OTError as exc:
        if recorder is not None:
            exc.run_id = exc.run_id or recorder.run_id
            recorder.failure(exc.category, exc.message, exc.details)
        raise
    except MemoryError as exc:
        error = ResourceExhaustedError(
            "Not enough memory to construct or update the transport arrays.",
            details={"exception": type(exc).__name__},
            run_id=recorder.run_id if recorder is not None else None,
        )
        if recorder is not None:
            recorder.failure(error.category, error.message, error.details)
        raise error from exc
    except Exception as exc:
        error = ComputationFailedError(
            "The numerical solver failed.",
            details={"exception": type(exc).__name__, "message": str(exc)},
            run_id=recorder.run_id if recorder is not None else None,
        )
        if recorder is not None:
            recorder.failure(error.category, error.message, error.details)
        raise error from exc


def _validate_inputs(
    a: Any,
    b: Any,
    cost: Any,
    epsilon: Any,
    max_iterations: Any,
    tolerance: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    try:
        a_vec = np.asarray(a, dtype=np.float64)
        b_vec = np.asarray(b, dtype=np.float64)
        c_mat = np.asarray(cost, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputQualityError("Marginals and cost must be numeric arrays.") from exc

    if a_vec.ndim != 1 or b_vec.ndim != 1:
        raise InputQualityError("Both marginals must be one-dimensional arrays.")
    if c_mat.ndim != 2:
        raise InputQualityError("The cost matrix must be two-dimensional.")
    if c_mat.shape != (a_vec.size, b_vec.size):
        raise InputQualityError(
            "The cost matrix shape must match the marginal lengths.",
            details={
                "marginal_lengths": [int(a_vec.size), int(b_vec.size)],
                "cost_shape": list(c_mat.shape),
            },
        )
    if not np.all(np.isfinite(a_vec)):
        raise InputQualityError("Row marginal contains NaN or infinite values.")
    if not np.all(np.isfinite(b_vec)):
        raise InputQualityError("Column marginal contains NaN or infinite values.")
    if not np.all(np.isfinite(c_mat)):
        raise InputQualityError("Cost matrix contains NaN or infinite values.")
    if np.any(a_vec < 0.0):
        raise InputQualityError("Row masses must be non-negative.")
    if np.any(b_vec < 0.0):
        raise InputQualityError("Column masses must be non-negative.")

    try:
        epsilon_value = float(epsilon)
        tolerance_value = float(tolerance)
        max_iterations_value = int(max_iterations)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputQualityError("Solver parameters must be finite numbers.") from exc
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise InputQualityError("epsilon must be a finite positive number.")
    if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise InputQualityError("tolerance must be a finite positive number.")
    if max_iterations_value <= 0 or max_iterations_value != max_iterations:
        raise InputQualityError("max_iterations must be a positive integer.")

    total_a = float(a_vec.sum())
    total_b = float(b_vec.sum())
    mass_scale = max(1.0, total_a, total_b)
    if abs(total_a - total_b) > 1e-12 * mass_scale:
        raise InputQualityError(
            "Total row and column masses must match.",
            details={
                "row_total": total_a,
                "column_total": total_b,
                "absolute_difference": abs(total_a - total_b),
            },
        )

    return (
        a_vec,
        b_vec,
        c_mat,
        epsilon_value,
        max_iterations_value,
        tolerance_value,
    )


def _relative_max_error(
    target: np.ndarray, observed: np.ndarray, tolerance: float, total_mass: float
) -> float:
    if not np.all(np.isfinite(observed)):
        return float("inf")
    absolute_error = np.abs(observed - target)
    threshold = tolerance * np.maximum(total_mass, target)
    threshold = np.maximum(threshold, np.finfo(np.float64).tiny)
    return float(np.max(absolute_error / threshold))


def _max_absolute_error(target: np.ndarray, observed: np.ndarray) -> float:
    if target.size == 0:
        return 0.0
    return float(np.max(np.abs(observed - target)))


def _within_tolerance(
    target: np.ndarray,
    observed: np.ndarray,
    tolerance: float,
    total_mass: float,
) -> bool:
    threshold = tolerance * np.maximum(total_mass, target)
    threshold = np.maximum(threshold, np.finfo(np.float64).tiny)
    return bool(np.all(np.abs(observed - target) <= threshold))


def _reconstruct_plan(
    log_kernel: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    repair_columns: bool,
) -> np.ndarray:
    """Construct a representable plan without global kernel exponentiation."""

    log_weights = log_kernel + f[:, None] + g[None, :]
    shifted = log_weights - np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(shifted)
    row_sum = weights.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(row_sum)) or np.any(row_sum <= 0.0):
        raise ComputationFailedError("Row-stable plan reconstruction underflowed.")
    plan = weights * (a[:, None] / row_sum)
    if repair_columns:
        _repair_column_marginals(plan, b)
    if np.any(plan < -np.finfo(np.float64).eps):
        raise ComputationFailedError("Plan repair produced a negative entry.")
    return np.maximum(plan, 0.0)


def _repair_column_marginals(plan: np.ndarray, b: np.ndarray) -> None:
    """Move mass within rows to remove a small reconstructed column residual."""

    epsilon_mass = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(b))))
    for _ in range(b.size + 1):
        residual = b - plan.sum(axis=0)
        if float(np.max(np.abs(residual))) <= epsilon_mass:
            return
        deficit_indices = np.flatnonzero(residual > epsilon_mass)
        surplus_indices = np.flatnonzero(residual < -epsilon_mass)
        if deficit_indices.size == 0 or surplus_indices.size == 0:
            return
        j = int(deficit_indices[np.argmax(residual[deficit_indices])])
        k = int(surplus_indices[np.argmin(residual[surplus_indices])])
        available = plan[:, k]
        total_available = float(available.sum())
        if total_available <= 0.0:
            return
        amount = min(float(residual[j]), -float(residual[k]), total_available)
        row_move = available * (amount / total_available)
        plan[:, k] -= row_move
        plan[:, j] += row_move


def _kl_penalty(
    plan: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    total_mass: float,
) -> float:
    if total_mass == 0.0:
        return 0.0
    outer = (a[:, None] * b[None, :]) / total_mass
    ratio = np.divide(
        plan,
        outer,
        out=np.zeros_like(plan),
        where=(plan > 0.0) & (outer > 0.0),
    )
    terms = np.where(plan > 0.0, plan * np.log(np.maximum(ratio, np.finfo(float).tiny)), 0.0)
    if not np.all(np.isfinite(terms)):
        raise ComputationFailedError("Entropy penalty contains a non-finite term.")
    return float(terms.sum())


def _feasibility_reason(
    a: np.ndarray,
    b: np.ndarray,
    row_marginals: np.ndarray,
    col_marginals: np.ndarray,
    tolerance: float,
    total_mass: float,
) -> str:
    row_ok = _within_tolerance(a, row_marginals, tolerance, total_mass)
    col_ok = _within_tolerance(b, col_marginals, tolerance, total_mass)
    if row_ok and col_ok:
        return "both reconstructed row and column marginals satisfy their tolerances"
    if not row_ok and not col_ok:
        return "both reconstructed row and column marginals exceed tolerance"
    if not row_ok:
        return "reconstructed row marginal exceeds tolerance"
    return "reconstructed column marginal exceeds tolerance"


def _zero_mass_result(
    shape: tuple[int, int],
    epsilon: float,
    max_iterations: int,
    tolerance: float,
) -> TransportResult:
    n, m = shape
    marginals = np.zeros(n, dtype=np.float64)
    zero_columns = np.zeros(m, dtype=np.float64)
    plan = np.zeros(shape, dtype=np.float64)
    return TransportResult(
        plan=plan,
        original_cost=0.0,
        regularized_cost=0.0,
        entropy_penalty=0.0,
        epsilon=epsilon,
        iterations=0,
        max_iterations=max_iterations,
        converged=True,
        feasible=True,
        stop_reason="zero_total_mass",
        row_error=0.0,
        column_error=0.0,
        row_marginals=marginals,
        column_marginals=zero_columns,
        diagnostics={
            "tolerance": tolerance,
            "total_mass": 0.0,
            "active_rows": [],
            "active_columns": [],
            "log_row_error_at_stop": 0.0,
            "log_column_error_at_stop": 0.0,
            "plan_mass": 0.0,
            "feasibility_reason": "all row and column masses are zero",
        },
    )


def _jsonable_arrays(
    a: np.ndarray, b: np.ndarray, cost: np.ndarray
) -> dict[str, list[float] | list[list[float]]]:
    return {"a": a.tolist(), "b": b.tolist(), "cost": cost.tolist()}


def _result_to_log(result: TransportResult) -> dict[str, Any]:
    return {
        "plan": result.plan.tolist(),
        "original_cost": result.original_cost,
        "regularized_cost": result.regularized_cost,
        "entropy_penalty": result.entropy_penalty,
        "iterations": result.iterations,
        "converged": result.converged,
        "feasible": result.feasible,
        "stop_reason": result.stop_reason,
        "row_error": result.row_error,
        "column_error": result.column_error,
        "row_marginals": result.row_marginals.tolist(),
        "column_marginals": result.column_marginals.tolist(),
        "diagnostics": result.diagnostics,
    }
