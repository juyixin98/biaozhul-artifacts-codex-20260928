"""Numerical evidence and identifiability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svdvals

from app.parsing import CompiledModel
from app.status import SolverError, Status


@dataclass(frozen=True)
class DerivativeCheck:
    max_absolute_error: float
    max_relative_error: float
    tolerance_absolute: float
    tolerance_relative: float
    passed: bool
    when: str


@dataclass(frozen=True)
class Identifiability:
    numerical_rank: int
    parameter_count: int
    singular_values: list[float]
    condition_number: float | None
    identifiable: bool
    parameter_correlation: list[list[float]]


def finite_or_illegal(values: np.ndarray, where: str) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        n_bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise SolverError(
            Status.ILLEGAL_FUNCTION_VALUE,
            f"non-finite function or residual at {where}",
            details={"nonfinite_count": n_bad, "where": where},
            http_status_code=422,
        )


def check_derivatives(
    model: CompiledModel,
    scaling,
    z: np.ndarray,
    x: np.ndarray,
    step: float,
    rtol: float,
    atol: float,
    when: str,
) -> DerivativeCheck:
    """Compare AD normalized Jacobian with normalized-coordinate differences."""

    analytic, _ = normalized_jacobian_with_x(model, scaling, z, x)
    lower, upper = scaling.internal_bounds()
    finite_difference = np.empty_like(analytic, dtype=float)

    for i in range(scaling.n):
        z_plus = z.copy()
        z_minus = z.copy()
        h_plus = min(step, max(step, upper[i] - z[i])) if np.isfinite(upper[i]) else step
        h_minus = min(step, max(step, z[i] - lower[i])) if np.isfinite(lower[i]) else step

        if z[i] + 2.0 * step <= upper[i] and z[i] - 2.0 * step >= lower[i]:
            z_plus[i] += step
            z_minus[i] -= step
            plus_value = model.evaluate(scaling.to_external(z_plus), x)
            minus_value = model.evaluate(scaling.to_external(z_minus), x)
            finite_or_illegal(plus_value, f"derivative plus {i} ({when})")
            finite_or_illegal(minus_value, f"derivative minus {i} ({when})")
            finite_difference[:, i] = (
                plus_value - minus_value
            ) / (2.0 * step)
        elif z[i] + step <= upper[i]:
            z_plus[i] += h_plus
            base_value = model.evaluate(scaling.to_external(z), x)
            plus_value = model.evaluate(scaling.to_external(z_plus), x)
            finite_or_illegal(plus_value, f"derivative forward {i} ({when})")
            finite_or_illegal(base_value, f"derivative base {i} ({when})")
            finite_difference[:, i] = (plus_value - base_value) / h_plus
        else:
            z_minus[i] -= h_minus
            base_value = model.evaluate(scaling.to_external(z), x)
            minus_value = model.evaluate(scaling.to_external(z_minus), x)
            finite_or_illegal(minus_value, f"derivative backward {i} ({when})")
            finite_or_illegal(base_value, f"derivative base {i} ({when})")
            finite_difference[:, i] = (base_value - minus_value) / h_minus

    error = np.abs(analytic - finite_difference)
    max_abs = float(np.max(error)) if error.size else 0.0
    scale = np.maximum(np.abs(analytic), np.abs(finite_difference))
    relative = error / np.maximum(scale, 1.0)
    max_rel = float(np.max(relative)) if relative.size else 0.0
    passed = np.all(error <= atol + rtol * np.maximum(np.abs(analytic), np.abs(finite_difference)))
    if not passed:
        raise SolverError(
            Status.DERIVATIVE_CHECK_FAILED,
            "automatic derivatives disagree with finite differences",
            details={
                "when": when,
                "max_absolute_error": max_abs,
                "max_relative_error": max_rel,
                "absolute_tolerance": atol,
                "relative_tolerance": rtol,
            },
            http_status_code=422,
        )
    return DerivativeCheck(max_abs, max_rel, atol, rtol, True, when)


def normalized_jacobian_with_x(
    model: CompiledModel, scaling, z: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta = scaling.to_external(z)
    physical_jacobian = model.jacobian(theta, x)
    dtheta_dz = scaling.external_gradient_from_internal(z)
    return physical_jacobian * dtheta_dz[np.newaxis, :], theta


def identifiability(jacobian: np.ndarray, rank_tolerance: float) -> Identifiability:
    _, p = jacobian.shape
    if p == 0:
        return Identifiability(0, 0, [], None, True, [])
    singular = np.asarray(svdvals(jacobian), dtype=float)
    cutoff = rank_tolerance * float(singular[0]) if singular.size and singular[0] > 0.0 else 0.0
    rank = int(np.count_nonzero(singular > cutoff))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else None

    normal = jacobian.T @ jacobian
    diagonal = np.sqrt(np.maximum(np.diag(normal), 0.0))
    correlation = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            denominator = diagonal[i] * diagonal[j]
            value = float(normal[i, j] / denominator) if denominator > 0.0 else 0.0
            correlation[i, j] = correlation[j, i] = np.clip(value, -1.0, 1.0)

    return Identifiability(
        numerical_rank=rank,
        parameter_count=p,
        singular_values=[float(v) for v in singular],
        condition_number=condition,
        identifiable=rank == p,
        parameter_correlation=correlation.tolist(),
    )
