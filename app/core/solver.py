from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.core.diagnostics import (
    DerivativeCheck,
    FitError,
    FitResult,
    FitStatus,
    Identifiability,
    IterationRecord,
)

ResidualFunction = Callable[[np.ndarray], np.ndarray]
JacobianFunction = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def default_scales(
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Explicit finite scales used in normalized parameter coordinates."""

    scales = np.maximum(1.0, np.abs(x0))
    finite_width = np.where(
        np.isfinite(lower) & np.isfinite(upper),
        np.maximum(upper - lower, np.finfo(float).eps),
        np.zeros_like(x0),
    )
    scales = np.where(scales > 1.0, scales, np.maximum(scales, finite_width / 2.0))
    return np.maximum(scales, np.sqrt(np.finfo(float).eps))


def _more_sorensen(
    gradient: np.ndarray,
    hessian: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, bool]:
    """Solve a trust-region subproblem by eigen decomposition and bisection."""

    if gradient.size == 0:
        return np.zeros(0), False
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    rotated_gradient = eigenvectors.T @ gradient
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    lambda_lower = max(0.0, -float(eigenvalues[0]) + tolerance)

    def step_for(lam: float) -> np.ndarray:
        denominators = eigenvalues + lam
        denominators = np.where(np.abs(denominators) < tolerance, np.inf, denominators)
        return eigenvectors @ (-rotated_gradient / denominators)

    newton_step = step_for(lambda_lower)
    if float(np.linalg.norm(newton_step)) <= radius and lambda_lower <= tolerance:
        return newton_step, False

    low = lambda_lower
    high = max(
        lambda_lower,
        float(np.linalg.norm(gradient)) / max(radius, np.finfo(float).eps),
    )
    for _ in range(80):
        high_step = step_for(high)
        if np.linalg.norm(high_step) <= radius:
            break
        high *= 2.0

    step = newton_step
    for _ in range(80):
        step = step_for(0.5 * (low + high))
        norm = float(np.linalg.norm(step))
        if abs(norm - radius) <= 1e-10 * radius or (high - low) <= 1e-14:
            break
        if norm > radius:
            low = 0.5 * (low + high)
        else:
            high = 0.5 * (low + high)

    if float(np.linalg.norm(step)) > 1.000001 * radius:
        step *= radius / float(np.linalg.norm(step))
    return step, True


def _active_and_stationarity(
    theta: np.ndarray,
    gradient: np.ndarray,
    scales: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    active_tol = 1e-10 * np.maximum(scales, 1.0)
    at_lower = (theta - lower) <= active_tol
    at_upper = (upper - theta) <= active_tol
    # Coordinates are z = theta / scales, so g_z = diag(scales) g_theta.
    scaled_gradient = scales * gradient

    active_lower = at_lower & (scaled_gradient >= -1e-12)
    active_upper = at_upper & (scaled_gradient <= 1e-12)
    active = active_lower | active_upper
    projected = np.where(active, 0.0, scales * gradient)
    return active_lower, active_upper, ~active, float(np.max(np.abs(projected)))


def identifiability_from_jacobian(
    jacobian: np.ndarray,
    scales: np.ndarray,
    *,
    rcond: float = 1e-10,
) -> Identifiability:
    scaled = jacobian * scales
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    threshold = (
        singular_values[0] * rcond * max(scaled.shape) if singular_values.size else 0.0
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    correlation: list[list[float | None]] = [
        [None for _ in range(scaled.shape[1])] for _ in range(scaled.shape[1])
    ]
    condition = None
    if rank == scaled.shape[1] and singular_values[-1] > 0:
        covariance = np.linalg.inv(scaled.T @ scaled)
        diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        outer = np.outer(diagonal, diagonal)
        raw_correlation = covariance / np.where(outer > 0, outer, np.nan)
        correlation = [
            [float(raw_correlation[i, j]) for j in range(scaled.shape[1])]
            for i in range(scaled.shape[1])
        ]
        condition = float(singular_values[0] / singular_values[-1])
    return Identifiability(
        rank=rank,
        parameter_count=scaled.shape[1],
        identifiable=rank == scaled.shape[1],
        singular_values=[float(v) for v in singular_values],
        condition_number=condition,
        parameter_correlation=correlation,
    )


def solve_least_squares(
    residual_fn: ResidualFunction,
    jacobian_fn: JacobianFunction,
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    scales: np.ndarray | None = None,
    max_iterations: int = 100,
    radius: float = 1.0,
    min_radius: float = 1e-12,
    gtol: float = 1e-8,
    xtol: float = 1e-10,
    ftol: float = 1e-10,
    max_invalid_trials: int = 20,
    derivative_check: DerivativeCheck | None = None,
    parameter_names: list[str] | None = None,
) -> FitResult:
    x0 = np.asarray(x0, dtype=float)
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    if x0.ndim != 1 or lower.shape != x0.shape or upper.shape != x0.shape:
        raise FitError(FitStatus.INVALID_INPUT, "x0 and bounds must have matching 1-D shapes")
    if np.any(lower >= upper):
        raise FitError(FitStatus.INVALID_INPUT, "each lower bound must be below its upper bound")
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise FitError(FitStatus.INVALID_INPUT, "initial point must lie inside bounds")

    parameter_names = parameter_names or [f"p{i}" for i in range(x0.size)]
    if scales is None:
        scales = default_scales(x0, lower, upper)
    else:
        scales = np.asarray(scales, dtype=float)
    if scales.shape != x0.shape or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise FitError(FitStatus.INVALID_INPUT, "scales must be finite, positive, and match x0")

    theta = x0.copy()
    try:
        residual, jacobian = jacobian_fn(theta)
    except FitError:
        raise
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FitError(
            FitStatus.INVALID_INITIAL_RESIDUAL,
            f"initial residual or Jacobian is not finite: {exc}",
        ) from exc

    residual = np.asarray(residual, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    nfev = 1
    njev = 1
    invalid_evaluations = 0
    iterations: list[IterationRecord] = []
    active_lower = np.zeros_like(theta, dtype=bool)
    active_upper = np.zeros_like(theta, dtype=bool)

    if residual.ndim != 1 or jacobian.shape != (residual.size, theta.size):
        raise FitError(
            FitStatus.INVALID_INPUT,
            "residual must be 1-D and the Jacobian must be m by n",
        )
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
        raise FitError(
            FitStatus.INVALID_INITIAL_RESIDUAL,
            "initial residual or Jacobian contains NaN or infinity",
        )

    def stationarity_label(lower_active: np.ndarray, upper_active: np.ndarray) -> str:
        active = lower_active | upper_active
        if not np.any(active):
            return "interior"
        if np.all(active):
            return "boundary"
        return "mixed"

    def make_result(
        status: FitStatus,
        message: str,
        reason: str,
        *,
        projected_gradient: float,
    ) -> FitResult:
        ident = identifiability_from_jacobian(jacobian, scales)
        success = status in {
            FitStatus.OPTIMAL,
            FitStatus.CONVERGED_COST_TOLERANCE,
            FitStatus.CONVERGED_STEP_SIZE,
        }
        if success and not ident.identifiable:
            status = FitStatus.UNIDENTIFIABLE
            success = False
            message = "stationary point reached, but the parameters are not identifiable"
            reason = "the scaled Jacobian is numerically rank deficient"
        return FitResult(
            status=status,
            success=success,
            message=message,
            reason=reason,
            parameter_names=parameter_names,
            x=theta.copy(),
            residuals=residual.copy(),
            cost=float(0.5 * np.dot(residual, residual)),
            jacobian=jacobian.copy(),
            projected_gradient_norm=float(projected_gradient),
            stationarity_type=stationarity_label(active_lower, active_upper),
            active_lower=active_lower.copy(),
            active_upper=active_upper.copy(),
            scales=scales.copy(),
            iterations=iterations,
            nfev=nfev,
            njev=njev,
            invalid_evaluations=invalid_evaluations,
            final_radius=float(radius),
            identifiability=ident,
            derivative_check=derivative_check,
        )

    cost = float(0.5 * np.dot(residual, residual))
    gradient = jacobian.T @ residual
    active_lower, active_upper, _free, pgnorm = _active_and_stationarity(
        theta, gradient, scales, lower, upper
    )
    if pgnorm <= gtol:
        return make_result(
            FitStatus.OPTIMAL,
            "projected first-order stationarity reached",
            f"projected gradient norm {pgnorm:.3e} <= {gtol:.3e}",
            projected_gradient=pgnorm,
        )

    for iteration in range(1, max_iterations + 1):
        gradient = jacobian.T @ residual
        active_lower, active_upper, free, pgnorm = _active_and_stationarity(
            theta, gradient, scales, lower, upper
        )
        if pgnorm <= gtol:
            return make_result(
                FitStatus.OPTIMAL,
                "projected first-order stationarity reached",
                f"projected gradient norm {pgnorm:.3e} <= {gtol:.3e}",
                projected_gradient=pgnorm,
            )

        free_indices = np.flatnonzero(free)
        scaled_jacobian_free = jacobian[:, free] * scales[free]
        scaled_gradient_free = scales[free] * gradient[free]
        hessian_free = scaled_jacobian_free.T @ scaled_jacobian_free
        p_free_scaled, _hit_radius = _more_sorensen(
            scaled_gradient_free, hessian_free, radius
        )

        p_scaled = np.zeros_like(theta)
        p_scaled[free_indices] = p_free_scaled
        p_theta = scales * p_scaled
        fractions = [1.0]
        for index in free_indices:
            direction = p_theta[index]
            if direction < 0.0 and np.isfinite(lower[index]):
                fractions.append(0.95 * (theta[index] - lower[index]) / -direction)
            if direction > 0.0 and np.isfinite(upper[index]):
                fractions.append(0.95 * (upper[index] - theta[index]) / direction)
        fraction = min(1.0, max(0.0, min(fractions)))
        p_scaled *= fraction
        p_theta *= fraction
        candidate = np.clip(theta + p_theta, lower, upper)
        step_scaled = (candidate - theta) / scales
        step_norm = float(np.linalg.norm(step_scaled))
        hit_tol = 1e-10 * np.maximum(scales, 1.0)
        bound_hit = (
            ((candidate - lower) <= hit_tol) | ((upper - candidate) <= hit_tol)
        ) & (np.abs(step_scaled) > 0.0)

        linear_change = jacobian @ (candidate - theta)
        predicted_reduction = float(
            -np.dot(gradient, candidate - theta) - 0.5 * np.dot(
                linear_change, linear_change
            )
        )

        lower_names = [parameter_names[i] for i in np.flatnonzero(active_lower)]
        upper_names = [parameter_names[i] for i in np.flatnonzero(active_upper)]
        hit_names = [parameter_names[i] for i in np.flatnonzero(bound_hit)]

        if step_norm <= np.finfo(float).eps or predicted_reduction <= 0.0:
            radius *= 0.25
            iterations.append(
                IterationRecord(
                    iteration=iteration,
                    accepted=False,
                    cost=cost,
                    actual_reduction=None,
                    predicted_reduction=predicted_reduction,
                    reduction_ratio=None,
                    radius=radius,
                    step_norm_scaled=step_norm,
                    projected_gradient_norm=pgnorm,
                    active_lower=lower_names,
                    active_upper=upper_names,
                    bound_hit=hit_names,
                    judgment="rejected: zero useful feasible decrease",
                )
            )
            if radius < min_radius:
                return make_result(
                    FitStatus.NON_CONVERGED_TRUST_REGION,
                    "trust region became too small to make useful progress",
                    f"radius {radius:.3e} below minimum {min_radius:.3e}",
                    projected_gradient=pgnorm,
                )
            continue

        try:
            trial_residual, trial_jacobian = jacobian_fn(candidate)
            nfev += 1
            njev += 1
        except FitError as exc:
            invalid_evaluations += 1
            radius *= 0.25
            iterations.append(
                IterationRecord(
                    iteration=iteration,
                    accepted=False,
                    cost=cost,
                    actual_reduction=None,
                    predicted_reduction=predicted_reduction,
                    reduction_ratio=None,
                    radius=radius,
                    step_norm_scaled=step_norm,
                    projected_gradient_norm=pgnorm,
                    active_lower=lower_names,
                    active_upper=upper_names,
                    bound_hit=hit_names,
                    judgment=f"rejected: illegal function value: {exc.message}",
                )
            )
            if invalid_evaluations > max_invalid_trials or radius < min_radius:
                return make_result(
                    FitStatus.INVALID_TRIAL_VALUE,
                    "too many illegal or non-finite trial function values",
                    f"{invalid_evaluations} invalid trial evaluations; last error: {exc.message}",
                    projected_gradient=pgnorm,
                )
            continue
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
            invalid_evaluations += 1
            radius *= 0.25
            iterations.append(
                IterationRecord(
                    iteration=iteration,
                    accepted=False,
                    cost=cost,
                    actual_reduction=None,
                    predicted_reduction=predicted_reduction,
                    reduction_ratio=None,
                    radius=radius,
                    step_norm_scaled=step_norm,
                    projected_gradient_norm=pgnorm,
                    active_lower=lower_names,
                    active_upper=upper_names,
                    bound_hit=hit_names,
                    judgment=f"rejected: illegal function value: {exc}",
                )
            )
            if invalid_evaluations > max_invalid_trials or radius < min_radius:
                return make_result(
                    FitStatus.INVALID_TRIAL_VALUE,
                    "too many illegal or non-finite trial function values",
                    f"{invalid_evaluations} invalid trial evaluations; last error: {exc}",
                    projected_gradient=pgnorm,
                )
            continue

        trial_residual = np.asarray(trial_residual, dtype=float)
        trial_jacobian = np.asarray(trial_jacobian, dtype=float)
        if not np.all(np.isfinite(trial_residual)) or not np.all(
            np.isfinite(trial_jacobian)
        ):
            invalid_evaluations += 1
            radius *= 0.25
            iterations.append(
                IterationRecord(
                    iteration=iteration,
                    accepted=False,
                    cost=cost,
                    actual_reduction=None,
                    predicted_reduction=predicted_reduction,
                    reduction_ratio=None,
                    radius=radius,
                    step_norm_scaled=step_norm,
                    projected_gradient_norm=pgnorm,
                    active_lower=lower_names,
                    active_upper=upper_names,
                    bound_hit=hit_names,
                    judgment="rejected: trial residual or Jacobian was NaN or infinity",
                )
            )
            if invalid_evaluations > max_invalid_trials or radius < min_radius:
                return make_result(
                    FitStatus.INVALID_TRIAL_VALUE,
                    "too many illegal or non-finite trial function values",
                    "trial residual or Jacobian contained NaN or infinity",
                    projected_gradient=pgnorm,
                )
            continue

        trial_cost = float(0.5 * np.dot(trial_residual, trial_residual))
        actual_reduction = cost - trial_cost
        reduction_ratio = actual_reduction / predicted_reduction
        accepted = reduction_ratio >= 1e-4

        if accepted:
            old_theta = theta.copy()
            theta = candidate
            residual = trial_residual
            jacobian = trial_jacobian
            old_cost = cost
            cost = trial_cost
            if reduction_ratio >= 0.75:
                radius = min(1e6, max(radius, 2.0 * step_norm))
            elif reduction_ratio < 0.25:
                radius = max(0.25 * radius, 0.5 * step_norm)
            judgment = (
                "accepted: actual decrease agrees with predicted model decrease"
            )
        else:
            old_theta = theta
            old_cost = cost
            radius = max(0.25 * radius, 0.5 * step_norm)
            judgment = "rejected: actual decrease was too small relative to prediction"

        iterations.append(
            IterationRecord(
                iteration=iteration,
                accepted=accepted,
                cost=cost,
                actual_reduction=actual_reduction,
                predicted_reduction=predicted_reduction,
                reduction_ratio=reduction_ratio,
                radius=radius,
                step_norm_scaled=step_norm,
                projected_gradient_norm=pgnorm,
                active_lower=lower_names,
                active_upper=upper_names,
                bound_hit=hit_names,
                judgment=judgment,
            )
        )

        if accepted and abs(old_cost - cost) <= ftol * max(1.0, old_cost):
            gradient = jacobian.T @ residual
            active_lower, active_upper, _free, pgnorm = _active_and_stationarity(
                theta, gradient, scales, lower, upper
            )
            return make_result(
                FitStatus.CONVERGED_COST_TOLERANCE,
                "relative objective change is below tolerance",
                f"cost change {abs(old_cost - cost):.3e} <= {ftol:.3e} * max(1, old cost)",
                projected_gradient=pgnorm,
            )

        normalized_move = float(np.linalg.norm((theta - old_theta) / scales))
        if accepted and normalized_move <= xtol:
            gradient = jacobian.T @ residual
            active_lower, active_upper, _free, pgnorm = _active_and_stationarity(
                theta, gradient, scales, lower, upper
            )
            return make_result(
                FitStatus.CONVERGED_STEP_SIZE,
                "normalized parameter change is below tolerance",
                f"scaled step {normalized_move:.3e} <= {xtol:.3e}",
                projected_gradient=pgnorm,
            )

        if not accepted and radius < min_radius:
            return make_result(
                FitStatus.NON_CONVERGED_TRUST_REGION,
                "trust region became too small to make useful progress",
                f"radius {radius:.3e} below minimum {min_radius:.3e}",
                projected_gradient=pgnorm,
            )

    gradient = jacobian.T @ residual
    active_lower, active_upper, _free, pgnorm = _active_and_stationarity(
        theta, gradient, scales, lower, upper
    )
    return make_result(
        FitStatus.NON_CONVERGED_MAX_ITERATIONS,
        "iteration limit was reached before stationarity",
        f"maximum iterations ({max_iterations}) exhausted; projected gradient {pgnorm:.3e}",
        projected_gradient=pgnorm,
    )
