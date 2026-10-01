"""Bounded nonlinear least-squares trust-region solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.linalg import svd

from .diagnostics import IterationRecord, RunLogger, SolveDiagnostics, package_versions
from .expression import ExpressionModel, ExpressionValidationError, IllegalFunctionError


class SolveStatus(str, Enum):
    CONVERGED_INTERIOR = "CONVERGED_INTERIOR"
    CONVERGED_BOUNDARY = "CONVERGED_BOUNDARY"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    NON_CONVERGED = "NON_CONVERGED"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    ILLEGAL_FUNCTION_VALUE = "ILLEGAL_FUNCTION_VALUE"
    DERIVATIVE_CHECK_FAILED = "DERIVATIVE_CHECK_FAILED"


@dataclass
class SolveOptions:
    max_iterations: int = 50
    tolerance: float = 1e-3
    initial_trust_radius: float = 0.3
    max_trust_radius: float = 10.0
    min_trust_radius: float = 1e-12
    derivative_check: bool = True
    derivative_tolerance: float = 5e-6
    rank_tolerance: float = 1e-10
    high_condition_warning: float = 1e4


@dataclass
class SolveResult:
    success: bool
    status: SolveStatus
    reason: str
    parameters: dict[str, float]
    residuals: list[float]
    cost: float
    iterations: int
    accepted_iterations: int
    stationary_type: str
    boundary_parameters: list[str]
    diagnostics: SolveDiagnostics

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "status": self.status.value,
            "reason": self.reason,
            "parameters": self.parameters,
            "residuals": self.residuals,
            "cost": self.cost,
            "iterations": self.iterations,
            "accepted_iterations": self.accepted_iterations,
            "stationary_type": self.stationary_type,
            "boundary_parameters": self.boundary_parameters,
            "diagnostics": self.diagnostics.to_dict(),
        }


def _finite_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ExpressionValidationError(f"{name} must be a non-empty one-dimensional finite numeric array")
    return array


def _sphere_intersection(q: np.ndarray, direction: np.ndarray, radius: float) -> float | None:
    a = float(direction @ direction)
    if a <= 0.0:
        return None
    b = 2.0 * float(q @ direction)
    c = float(q @ q) - radius**2
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return 0.0 if c <= 0.0 else None
    root_plus = (-b + math.sqrt(max(0.0, discriminant))) / (2.0 * a)
    root_minus = (-b - math.sqrt(max(0.0, discriminant))) / (2.0 * a)
    roots = [root for root in (root_minus, root_plus) if root >= -1e-12]
    return max(0.0, min(roots)) if roots else None


def _box_intersection(
    q: np.ndarray,
    direction: np.ndarray,
    z: np.ndarray,
    lower_z: np.ndarray,
    upper_z: np.ndarray,
) -> float | None:
    current = z + q
    ratios: list[float] = []
    positive = direction > 0.0
    negative = direction < 0.0
    if np.any(positive):
        ratios.extend(((upper_z - current)[positive] / direction[positive]).tolist())
    if np.any(negative):
        ratios.extend(((lower_z - current)[negative] / direction[negative]).tolist())
    valid = [value for value in ratios if value >= -1e-12]
    return max(0.0, float(min(valid))) if valid else None


def _clip_step(
    q: np.ndarray,
    z: np.ndarray,
    lower_z: np.ndarray,
    upper_z: np.ndarray,
    radius: float,
) -> np.ndarray:
    clipped = np.minimum(np.maximum(q, lower_z - z), upper_z - z)
    norm = float(np.linalg.norm(clipped))
    if norm > radius and norm > 0.0:
        clipped *= radius / norm
    return clipped


def active_steihaug(
    gradient: np.ndarray,
    hessian: np.ndarray,
    radius: float,
    z: np.ndarray,
    lower_z: np.ndarray,
    upper_z: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Solve the boxed Gauss-Newton quadratic in scaled coordinates."""
    n = gradient.size
    tol = 1e-10
    blocked = ((z - lower_z <= tol) & (gradient >= 0.0)) | ((upper_z - z <= tol) & (gradient <= 0.0))
    free = ~blocked
    if not np.any(free):
        return np.zeros(n), False

    q = np.zeros(n)
    residual = gradient.copy()
    diagonal = np.maximum(np.diag(hessian), 1e-14)
    preconditioned = np.zeros(n)
    preconditioned[free] = residual[free] / diagonal[free]
    direction = -preconditioned
    rr = float(residual[free] @ preconditioned[free])
    hit_boundary = False

    for _ in range(max(20, 4 * n)):
        h_direction = hessian @ direction
        curvature = float(direction @ h_direction)
        if curvature < -1e-12 * max(1.0, float(direction @ direction)):
            if rr <= 1e-20:
                hit_boundary = float(np.linalg.norm(q)) >= radius * (1.0 - 1e-10)
                return _clip_step(q, z, lower_z, upper_z, radius), hit_boundary
            sphere_tau = _sphere_intersection(q, direction, radius)
            box_tau = _box_intersection(q, direction, z, lower_z, upper_z)
            candidates = [tau for tau in (sphere_tau, box_tau) if tau is not None]
            q += min(candidates) * direction
            return _clip_step(q, z, lower_z, upper_z, radius), True
        if abs(curvature) <= 1e-12 * max(1.0, float(direction @ direction)) and rr <= 1e-20:
            hit_boundary = float(np.linalg.norm(q)) >= radius * (1.0 - 1e-10)
            return _clip_step(q, z, lower_z, upper_z, radius), hit_boundary

        alpha = rr / curvature
        sphere_tau = _sphere_intersection(q, direction, radius)
        box_tau = _box_intersection(q, direction, z, lower_z, upper_z)
        candidates = [tau for tau in (alpha, sphere_tau, box_tau) if tau is not None]
        tau = min(candidates)
        if tau < alpha - 1e-12 * max(1.0, abs(alpha)):
            q += tau * direction
            hit_boundary = True
            break

        q += alpha * direction
        residual = residual + alpha * h_direction
        residual[~free] = 0.0
        preconditioned = np.zeros(n)
        preconditioned[free] = residual[free] / diagonal[free]
        rr_new = float(residual[free] @ preconditioned[free])
        if rr_new <= 1e-24:
            break
        if np.linalg.norm(q) >= radius * (1.0 - 1e-10):
            hit_boundary = True
            break
        beta = rr_new / max(rr, 1e-30)
        old_direction = direction.copy()
        direction = -preconditioned + beta * old_direction
        direction[~free] = 0.0
        rr = rr_new

    return _clip_step(q, z, lower_z, upper_z, radius), hit_boundary


def projected_scaled_gradient(
    gradient_scaled: np.ndarray,
    z: np.ndarray,
    lower_z: np.ndarray,
    upper_z: np.ndarray,
) -> np.ndarray:
    projected = gradient_scaled.copy()
    at_lower = z - lower_z <= 1e-9
    at_upper = upper_z - z <= 1e-9
    projected[at_lower & (gradient_scaled >= 0.0)] = 0.0
    projected[at_upper & (gradient_scaled <= 0.0)] = 0.0
    return projected


def finite_difference_check(
    model: ExpressionModel,
    x: np.ndarray,
    theta: np.ndarray,
    names: list[str],
    scales: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Check analytic AD columns against independent finite differences."""
    values, analytic = model.evaluate(x, dict(zip(names, theta)))
    max_error = 0.0
    step = math.sqrt(np.finfo(float).eps)
    for j, name in enumerate(names):
        h = step * max(abs(theta[j]), scales[j], 1.0)
        params = dict(zip(names, theta))
        can_forward = theta[j] + h <= upper[j]
        can_backward = theta[j] - h >= lower[j]
        if can_forward and can_backward:
            params[name] = theta[j] + h
            plus = model.evaluate(x, params)[0]
            params[name] = theta[j] - h
            minus = model.evaluate(x, params)[0]
            numeric = (plus - minus) / (2.0 * h)
        elif can_forward:
            params[name] = theta[j] + h
            plus = model.evaluate(x, params)[0]
            numeric = (plus - values) / h
        elif can_backward:
            params[name] = theta[j] - h
            minus = model.evaluate(x, params)[0]
            numeric = (values - minus) / h
        else:
            raise IllegalFunctionError(f"No finite-difference perturbation is feasible for {name}")
        norm_scale = max(1.0, float(np.linalg.norm(analytic[:, j])), float(np.linalg.norm(numeric)))
        max_error = max(max_error, float(np.max(np.abs(analytic[:, j] - numeric)) / norm_scale))
    return max_error


def solve_nonlinear_least_squares(
    expression: str,
    parameter_names: list[str],
    x: np.ndarray,
    y: np.ndarray,
    initial: dict[str, float],
    lower: dict[str, float],
    upper: dict[str, float],
    scales: dict[str, float] | None = None,
    options: SolveOptions | None = None,
    logger: RunLogger | None = None,
    run_id: str | None = None,
) -> SolveResult:
    options = options or SolveOptions()
    owns_logger = logger is None
    logger = logger or RunLogger(run_id=run_id)
    run_id = logger.run_id
    names = list(parameter_names)

    x_array = _finite_array(x, "x")
    y_array = _finite_array(y, "y")
    if x_array.shape != y_array.shape:
        raise ExpressionValidationError("x and y must have the same length")
    theta0 = np.array([float(initial[name]) for name in names], dtype=float)
    lower_theta = np.array([float(lower[name]) for name in names], dtype=float)
    upper_theta = np.array([float(upper[name]) for name in names], dtype=float)
    if not np.all(np.isfinite(theta0)) or not np.all(np.isfinite(lower_theta)):
        raise ExpressionValidationError("Initial values and bounds must be finite")
    if not np.all(np.isfinite(upper_theta)):
        raise ExpressionValidationError("Initial values and bounds must be finite")
    if np.any(lower_theta >= upper_theta):
        raise ExpressionValidationError("Each lower bound must be strictly below its upper bound")
    if np.any(theta0 < lower_theta) or np.any(theta0 > upper_theta):
        raise ExpressionValidationError("Initial values must satisfy lower <= initial <= upper")

    scale_values = np.ones(len(names))
    scale_source: dict[str, str] = {}
    for j, name in enumerate(names):
        if scales and name in scales:
            value = float(scales[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ExpressionValidationError(f"Scale for {name} must be a positive finite number")
            scale_values[j] = value
            scale_source[name] = "provided"
        else:
            scale_values[j] = max(abs(theta0[j]), float(upper_theta[j] - lower_theta[j]), 1.0)
            scale_source[name] = "automatic:max(|initial|,bound_span,1)"

    model = ExpressionModel(expression, names)
    diagnostics = SolveDiagnostics(
        run_id=run_id,
        expression=expression,
        parameter_names=names,
        parameter_scales=scale_values.tolist(),
        scale_source=scale_source,
        observations=int(x_array.size),
        max_iterations=options.max_iterations,
        tolerance=options.tolerance,
        versions=package_versions(),
    )
    logger.event(
        "solve_started",
        expression=expression,
        parameters=names,
        observations=int(x_array.size),
        initial=theta0,
        lower=lower_theta,
        upper=upper_theta,
        scales=scale_values,
    )

    def fail(
        status: SolveStatus,
        reason: str,
        theta: np.ndarray | None = None,
        iterations: int = 0,
        accepted_iterations: int = 0,
    ) -> SolveResult:
        diagnostics.status = status.value
        diagnostics.reason = reason
        logger.event("solve_finished", success=False, status=status.value, reason=reason)
        if owns_logger:
            for handler in logger._logger.handlers:
                handler.close()
                logger._logger.removeHandler(handler)
        values = theta if theta is not None else np.full(len(names), np.nan)
        try:
            failed_prediction, _ = model.evaluate(x_array, dict(zip(names, values)))
            failed_residual = (failed_prediction - y_array).tolist()
            failed_cost = 0.5 * float((failed_prediction - y_array) @ (failed_prediction - y_array))
        except Exception:
            failed_residual = []
            failed_cost = float("nan")
        return SolveResult(
            False,
            status,
            reason,
            dict(zip(names, (float(v) for v in values))),
            failed_residual,
            failed_cost,
            iterations,
            accepted_iterations,
            "none",
            [],
            diagnostics,
        )

    try:
        predicted, jacobian_theta = model.evaluate(x_array, dict(zip(names, theta0)))
        residual = predicted - y_array
    except IllegalFunctionError as exc:
        return fail(SolveStatus.ILLEGAL_FUNCTION_VALUE, f"Illegal value at initial point: {exc}", theta0)

    if options.derivative_check:
        try:
            fd_error = finite_difference_check(
                model, x_array, theta0, names, scale_values, lower_theta, upper_theta
            )
            diagnostics.finite_difference_max_error = fd_error
            logger.event(
                "derivative_check",
                max_relative_error=fd_error,
                passed=fd_error <= options.derivative_tolerance,
            )
            if fd_error > options.derivative_tolerance:
                return fail(
                    SolveStatus.DERIVATIVE_CHECK_FAILED,
                    f"AD/finite-difference mismatch {fd_error:.3e} exceeds tolerance",
                    theta0,
                )
        except IllegalFunctionError as exc:
            return fail(SolveStatus.ILLEGAL_FUNCTION_VALUE, f"Derivative check hit an illegal value: {exc}", theta0)

    # theta = S z; one unit in z corresponds to one declared parameter scale.
    lower_z = lower_theta / scale_values
    upper_z = upper_theta / scale_values
    z = theta0 / scale_values
    if not np.all((z >= lower_z - 1e-10) & (z <= upper_z + 1e-10)):
        raise ExpressionValidationError("Scaled initial point lies outside its scaled bounds")
    z = np.minimum(np.maximum(z, lower_z), upper_z)

    scaled_jac = jacobian_theta * scale_values
    _, singular_values, vt = svd(scaled_jac, full_matrices=True)
    diagnostics.singular_values = singular_values.tolist()
    if singular_values.size == 0:
        diagnostics.rank = 0
        diagnostics.condition_number = float("inf")
        diagnostics.nullspace = vt.tolist()
        return fail(SolveStatus.UNIDENTIFIABLE, "Fewer observations than parameters", theta0)
    threshold = options.rank_tolerance * max(scaled_jac.shape) * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    diagnostics.rank = rank
    diagnostics.condition_number = condition
    logger.event("identifiability", rank=rank, required_rank=len(names), condition_number=condition)
    if rank < len(names):
        diagnostics.nullspace = vt[rank:].tolist()
        return fail(
            SolveStatus.UNIDENTIFIABLE,
            f"Scaled Jacobian rank {rank} is below required rank {len(names)}",
            theta0,
        )

    cost = 0.5 * float(residual @ residual)
    radius = min(options.initial_trust_radius, options.max_trust_radius)
    accepted_count = 0
    theta = theta0.copy()
    grad_scaled = scale_values * (jacobian_theta.T @ residual)

    def boundary_names(current_z: np.ndarray) -> list[str]:
        return [
            name
            for j, name in enumerate(names)
            if abs(current_z[j] - lower_z[j]) <= 1e-8 or abs(current_z[j] - upper_z[j]) <= 1e-8
        ]

    def succeed(
        status: SolveStatus,
        reason: str,
        current_theta: np.ndarray,
        current_residual: np.ndarray,
        current_cost: float,
        iterations: int,
        current_z: np.ndarray,
    ) -> SolveResult:
        diagnostics.status = status.value
        diagnostics.reason = reason
        stationary_type = "boundary" if status == SolveStatus.CONVERGED_BOUNDARY else "interior"
        logger.event(
            "solve_finished",
            success=True,
            status=status.value,
            reason=reason,
            cost=current_cost,
            iterations=iterations,
            accepted_iterations=accepted_count,
        )
        if owns_logger:
            for handler in logger._logger.handlers:
                handler.close()
                logger._logger.removeHandler(handler)
        return SolveResult(
            True,
            status,
            reason,
            dict(zip(names, (float(v) for v in current_theta))),
            current_residual.tolist(),
            current_cost,
            iterations,
            accepted_count,
            stationary_type,
            boundary_names(current_z),
            diagnostics,
        )

    initial_projection = projected_scaled_gradient(grad_scaled, z, lower_z, upper_z)
    initial_boundary = boundary_names(z)
    if float(np.linalg.norm(initial_projection)) <= options.tolerance:
        status = SolveStatus.CONVERGED_BOUNDARY if initial_boundary else SolveStatus.CONVERGED_INTERIOR
        return succeed(status, "Initial point satisfies first-order stationarity", theta0, residual, cost, 0, z)

    for iteration in range(1, options.max_iterations + 1):
        scaled_jac = jacobian_theta * scale_values
        hessian_scaled = scaled_jac.T @ scaled_jac
        step_z, hit_trust_boundary = active_steihaug(grad_scaled, hessian_scaled, radius, z, lower_z, upper_z)
        step_norm = float(np.linalg.norm(step_z))
        predicted_reduction = -float(
            grad_scaled @ step_z + 0.5 * step_z @ hessian_scaled @ step_z
        )
        current_projection = projected_scaled_gradient(grad_scaled, z, lower_z, upper_z)
        current_projection_norm = float(np.linalg.norm(current_projection))
        if step_norm <= 1e-14 or predicted_reduction <= 1e-14:
            if current_projection_norm <= options.tolerance:
                current_boundary = boundary_names(z)
                status = (
                    SolveStatus.CONVERGED_BOUNDARY
                    if current_boundary
                    else SolveStatus.CONVERGED_INTERIOR
                )
                return succeed(
                    status,
                    "No numerically useful trust-region step remains and projected gradient is below tolerance",
                    theta,
                    residual,
                    cost,
                    iteration - 1,
                    z,
                )
            return fail(
                SolveStatus.NON_CONVERGED,
                "Subproblem produced no useful descent",
                theta,
                iteration - 1,
                accepted_count,
            )

        candidate_z = np.minimum(np.maximum(z + step_z, lower_z), upper_z)
        candidate_theta = scale_values * candidate_z
        active_lower = [
            name for name, value, bound in zip(names, candidate_z, lower_z) if abs(value - bound) <= 1e-8
        ]
        active_upper = [
            name for name, value, bound in zip(names, candidate_z, upper_z) if abs(value - bound) <= 1e-8
        ]

        trial = 0
        while True:
            trial += 1
            try:
                candidate_prediction, candidate_jac = model.evaluate(
                    x_array, dict(zip(names, candidate_theta))
                )
                candidate_residual = candidate_prediction - y_array
                candidate_cost = 0.5 * float(candidate_residual @ candidate_residual)
                if not np.all(np.isfinite(candidate_residual)) or not np.all(np.isfinite(candidate_jac)):
                    raise IllegalFunctionError("Non-finite candidate values")
            except IllegalFunctionError as exc:
                logger.event(
                    "illegal_candidate",
                    iteration=iteration,
                    trial=trial,
                    reason=str(exc),
                    trust_radius=radius,
                )
                radius *= 0.25
                if radius < options.min_trust_radius:
                    return fail(
                        SolveStatus.ILLEGAL_FUNCTION_VALUE,
                        "All acceptable steps lead outside the function domain",
                        theta,
                        iteration - 1,
                        accepted_count,
                    )
                step_z, hit_trust_boundary = active_steihaug(
                    grad_scaled, hessian_scaled, radius, z, lower_z, upper_z
                )
                step_norm = float(np.linalg.norm(step_z))
                candidate_z = np.minimum(np.maximum(z + step_z, lower_z), upper_z)
                candidate_theta = scale_values * candidate_z
                continue

            actual_reduction = cost - candidate_cost
            reduction_ratio = actual_reduction / predicted_reduction if predicted_reduction > 0 else float("nan")
            accepted = reduction_ratio >= 0.1
            candidate_grad = scale_values * (candidate_jac.T @ candidate_residual)
            projection_norm = float(
                np.linalg.norm(
                    projected_scaled_gradient(candidate_grad, candidate_z, lower_z, upper_z)
                )
            )
            on_boundary = bool(active_lower or active_upper)
            interior_stationary = projection_norm <= options.tolerance and not on_boundary
            boundary_stationary = projection_norm <= options.tolerance and on_boundary

            diagnostics.append(
                IterationRecord(
                    iteration=iteration,
                    cost=candidate_cost,
                    scaled_projected_gradient_norm=projection_norm,
                    trust_radius=radius,
                    step_norm=step_norm,
                    predicted_reduction=predicted_reduction,
                    actual_reduction=actual_reduction,
                    reduction_ratio=reduction_ratio,
                    accepted=accepted,
                    decision="accepted" if accepted else "rejected_shrink_radius",
                    active_lower=active_lower,
                    active_upper=active_upper,
                    interior_stationary=interior_stationary,
                    boundary_stationary=boundary_stationary,
                )
            )
            logger.event(
                "iteration",
                iteration=iteration,
                trial=trial,
                cost=candidate_cost,
                projected_gradient_norm=projection_norm,
                trust_radius=radius,
                step_norm=step_norm,
                predicted_reduction=predicted_reduction,
                actual_reduction=actual_reduction,
                reduction_ratio=reduction_ratio,
                accepted=accepted,
                active_lower=active_lower,
                active_upper=active_upper,
            )

            if not accepted:
                radius *= 0.25
                if radius < options.min_trust_radius:
                    return fail(
                        SolveStatus.NON_CONVERGED,
                        "Trust radius collapsed while rejecting the local quadratic step",
                        theta,
                        iteration - 1,
                        accepted_count,
                    )
                step_z, hit_trust_boundary = active_steihaug(
                    grad_scaled, hessian_scaled, radius, z, lower_z, upper_z
                )
                step_norm = float(np.linalg.norm(step_z))
                if trial >= 20 or step_norm <= 1e-14:
                    return fail(
                        SolveStatus.NON_CONVERGED,
                        "No progress after repeated radius reductions",
                        theta,
                        iteration - 1,
                        accepted_count,
                    )
                candidate_z = np.minimum(np.maximum(z + step_z, lower_z), upper_z)
                candidate_theta = scale_values * candidate_z
                predicted_reduction = -float(
                    grad_scaled @ step_z + 0.5 * step_z @ hessian_scaled @ step_z
                )
                continue

            accepted_count += 1
            theta = candidate_theta
            z = candidate_z
            residual = candidate_residual
            jacobian_theta = candidate_jac
            grad_scaled = candidate_grad
            cost = candidate_cost

            if projection_norm <= options.tolerance:
                status = (
                    SolveStatus.CONVERGED_BOUNDARY
                    if on_boundary
                    else SolveStatus.CONVERGED_INTERIOR
                )
                return succeed(
                    status,
                    "Scaled projected gradient satisfies first-order stationarity",
                    theta,
                    residual,
                    cost,
                    iteration,
                    z,
                )

            if reduction_ratio > 0.75 and hit_trust_boundary and step_norm >= 0.9 * radius:
                radius = min(options.max_trust_radius, 2.0 * radius)
            elif reduction_ratio < 0.25:
                radius = max(options.min_trust_radius, 0.5 * radius)
            break

    return fail(
        SolveStatus.MAX_ITERATIONS,
        f"Maximum {options.max_iterations} iterations reached without stationarity",
        theta,
        options.max_iterations,
        accepted_count,
    )
