"""Cyclic coordinate descent for a Lasso regularization path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .preprocessing import PreparedData


class SolverError(ValueError):
    """Raised for solver parameters that do not define a Lasso path."""


@dataclass(frozen=True)
class LambdaDiagnostic:
    lambda_value: float
    converged: bool
    iterations: int
    kkt_violation: float
    duality_gap: float
    primal_objective: float
    support_size: int
    status: str
    reason: str


@dataclass(frozen=True)
class LassoPathResult:
    lambdas: np.ndarray
    coefficients: np.ndarray
    standardized_coefficients: np.ndarray
    intercepts: np.ndarray
    support: np.ndarray
    diagnostics: list[LambdaDiagnostic]
    converged: bool
    lambda_order: str
    warm_start: bool


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def _as_positive_array(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise SolverError(f"{name} must contain at least one value")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise SolverError(f"{name} must contain finite, strictly positive values")
    return array


def _correlations(prepared: PreparedData, residual: np.ndarray) -> np.ndarray:
    if prepared.is_sparse:
        gradient = np.asarray(prepared.x.T @ residual).ravel() / prepared.n_samples
        # Effective sparse columns are centered, and residual stays centered.
        # Explicitly neutralize rejected/filled zero-variance columns.
        gradient[prepared.zero_variance] = 0.0
        return gradient
    return prepared.x.T @ residual / prepared.n_samples


def _kkt_and_gap(
    prepared: PreparedData,
    coef: np.ndarray,
    residual: np.ndarray,
    penalty: float,
) -> tuple[float, float, float]:
    gradient = _correlations(prepared, residual)
    active = coef != 0.0

    active_violation = np.abs(gradient[active] - penalty * np.sign(coef[active]))
    inactive_violation = np.maximum(np.abs(gradient[~active]) - penalty, 0.0)
    violations = np.concatenate((active_violation, inactive_violation))
    kkt_violation = float(violations.max()) if violations.size else 0.0

    residual_sq = float(np.dot(residual, residual))
    loss = residual_sq / (2.0 * prepared.n_samples)
    l1_norm = float(np.sum(np.abs(coef)))
    primal = loss + penalty * l1_norm

    max_correlation = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    dual_scale = 1.0
    if max_correlation > penalty:
        dual_scale = penalty / max_correlation
    theta = dual_scale * residual / prepared.n_samples
    dual = -0.5 * prepared.n_samples * float(np.dot(theta, theta)) + float(
        np.dot(theta, prepared.y_centered)
    )
    duality_gap = max(primal - dual, 0.0)
    return kkt_violation, duality_gap, primal


def _initial_residual(
    prepared: PreparedData, coef: np.ndarray
) -> np.ndarray:
    """Build residual for standardized sparse columns without densifying X."""
    if not prepared.is_sparse:
        return prepared.y_centered - prepared.matvec(coef)

    # Stored sparse X is X/scales. The effective design is X/scales - mean/scales.
    offset = np.dot(prepared.x_means / prepared.scales, coef)
    return prepared.y_centered - prepared.matvec(coef) + offset


def _coordinate_update(
    prepared: PreparedData,
    coef: np.ndarray,
    residual: np.ndarray,
    j: int,
    penalty: float,
) -> None:
    old = coef[j]
    column = prepared.column_dot(j)
    denominator = prepared.coordinate_denominator(j)

    if sparse.issparse(column):
        old_column = np.zeros_like(residual)
        if column.nnz:
            old_column[column.indices] = column.data
        effective_old_column = (
            old_column - prepared.x_means[j] / prepared.scales[j]
        )
        partial = residual + old * effective_old_column
        correlation = (
            float(np.dot(column.data, partial[column.indices]))
            if column.nnz
            else 0.0
        )
        correlation -= (
            prepared.x_means[j] / prepared.scales[j] * float(partial.sum())
        )
        correlation /= prepared.n_samples
    else:
        partial = residual + old * column
        correlation = float(np.dot(column, partial)) / prepared.n_samples

    new = (
        _soft_threshold(correlation, penalty) / denominator
    )
    delta = new - old
    if delta == 0.0:
        return

    coef[j] = new
    if prepared.is_sparse:
        # Add the omitted frozen column mean while preserving sparse X.
        residual -= delta * old_column
        residual += delta * prepared.x_means[j] / prepared.scales[j]
    else:
        residual -= delta * column


def default_lambda_grid(
    prepared: PreparedData,
    *,
    n_alphas: int = 100,
    lambda_min_ratio: float | None = None,
) -> np.ndarray:
    """Construct the standard log-spaced lambda grid."""

    if n_alphas < 2:
        raise SolverError("n_alphas must be at least two")
    max_correlation = float(
        np.max(np.abs(_correlations(prepared, prepared.y_centered)))
    )
    if max_correlation <= 0.0:
        raise SolverError(
            "lambda_max is zero; centered response has no correlation with X"
        )

    if lambda_min_ratio is None:
        lambda_min_ratio = (
            1e-3 if prepared.n_samples < prepared.n_features else 1e-4
        )
    if not 0.0 < lambda_min_ratio <= 1.0:
        raise SolverError("lambda_min_ratio must be in the interval (0, 1]")

    return np.logspace(
        np.log10(max_correlation * lambda_min_ratio),
        np.log10(max_correlation),
        num=n_alphas,
    )


def lasso_path(
    prepared: PreparedData,
    *,
    lambdas: np.ndarray | list[float] | None = None,
    n_alphas: int = 100,
    lambda_min_ratio: float | None = None,
    lambda_order: str = "descending",
    warm_start: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> LassoPathResult:
    """Fit a Lasso path with cyclic coordinate descent.

    The optimization problem is
    ``0.5 * mean((y - intercept - Z @ beta)^2) + lambda * ||beta||_1``,
    where ``Z`` is built from the frozen centered/scaled features. Coefficients
    returned by this function are restored to the original feature scale.
    """

    if lambda_order not in {"descending", "ascending"}:
        raise SolverError("lambda_order must be 'descending' or 'ascending'")
    if max_iter < 1:
        raise SolverError("max_iter must be at least one")
    if tol <= 0.0 or not np.isfinite(tol):
        raise SolverError("tol must be a finite, strictly positive number")

    grid = (
        _as_positive_array("lambdas", lambdas)
        if lambdas is not None
        else default_lambda_grid(
            prepared,
            n_alphas=n_alphas,
            lambda_min_ratio=lambda_min_ratio,
        )
    )
    grid.sort()
    if lambda_order == "descending":
        grid = grid[::-1]

    n_features = prepared.n_features
    coef = np.zeros(n_features, dtype=float)
    residual = prepared.y_centered.copy()
    standardized_path = []
    diagnostics: list[LambdaDiagnostic] = []

    for penalty in grid:
        if warm_start and standardized_path:
            # ``coef`` and ``residual`` already describe the previous solution.
            pass
        else:
            coef = np.zeros(n_features, dtype=float)
            residual = _initial_residual(prepared, coef)

        converged = False
        iterations = 0
        kkt_violation = np.inf
        duality_gap = np.inf
        primal_objective = np.inf

        for iterations in range(1, max_iter + 1):
            for j in range(n_features):
                if prepared.zero_variance[j]:
                    continue
                _coordinate_update(
                    prepared, coef, residual, j, float(penalty)
                )

            kkt_violation, duality_gap, primal_objective = _kkt_and_gap(
                prepared, coef, residual, float(penalty)
            )
            if kkt_violation <= tol:
                converged = True
                break

        status = "converged" if converged else "max_iterations"
        reason = (
            f"KKT violation {kkt_violation:.3e} <= tolerance {tol:.3e}"
            if converged
            else f"KKT violation {kkt_violation:.3e} > tolerance {tol:.3e} "
            f"after {max_iter} coordinate-descent passes"
        )
        support_size = int(np.count_nonzero(coef))
        diagnostics.append(
            LambdaDiagnostic(
                lambda_value=float(penalty),
                converged=converged,
                iterations=iterations,
                kkt_violation=float(kkt_violation),
                duality_gap=float(duality_gap),
                primal_objective=float(primal_objective),
                support_size=support_size,
                status=status,
                reason=reason,
            )
        )
        standardized_path.append(coef.copy())

    standardized_array = np.asarray(standardized_path)
    original_coefficients = np.vstack(
        [prepared.to_original_coefficients(row) for row in standardized_array]
    )
    intercepts = np.asarray(
        [prepared.original_intercept(row) for row in standardized_array]
    )
    support = np.not_equal(original_coefficients, 0.0)

    return LassoPathResult(
        lambdas=grid,
        coefficients=original_coefficients,
        standardized_coefficients=standardized_array,
        intercepts=intercepts,
        support=support,
        diagnostics=diagnostics,
        converged=all(item.converged for item in diagnostics),
        lambda_order=lambda_order,
        warm_start=warm_start,
    )
