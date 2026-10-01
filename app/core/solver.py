"""Bounded nonlinear least-squares trust-region driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.config import SolverConfig
from app.core.diagnostics import (
    check_derivatives,
    finite_or_illegal,
    identifiability,
    normalized_jacobian_with_x,
)
from app.core.scaling import ParameterScaling
from app.core.trust_region import solve_boundary_trust_region
from app.observability import RunLogger
from app.parsing import CompiledModel
from app.status import SolverError, Status


@dataclass
class SolveResult:
    run_id: str
    status: Status
    message: str
    expression: str
    parameter_names: list[str]
    parameters: list[float]
    predicted: list[float]
    residuals: list[float]
    residual_sum_squares: float
    max_abs_residual: float
    iterations: int
    trust_radius: float
    projected_gradient_norm: float
    stationarity_type: str | None
    active_bounds: dict[str, str]
    identifiability: dict[str, Any]
    derivative_checks: list[dict[str, Any]]
    iteration_history: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def solve_least_squares(
    *,
    model: CompiledModel,
    x: np.ndarray,
    y: np.ndarray,
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
    config: SolverConfig,
    logger: RunLogger,
) -> SolveResult:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    initial_array = np.asarray(initial, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    scales_array = np.asarray(scales, dtype=float)

    _validate_arrays(model, x_array, y_array, initial_array, lower_array, upper_array, scales_array)
    try:
        scaling = ParameterScaling.create(lower_array, upper_array, scales_array)
    except ValueError as exc:
        raise SolverError(Status.INVALID_REQUEST, str(exc), http_status_code=400) from exc

    logger.log_start(model.expression, len(x_array), model.parameter_names)
    z = scaling.to_internal(initial_array)
    lower_z, upper_z = scaling.internal_bounds()
    z = np.minimum(np.maximum(z, lower_z), upper_z)

    theta, prediction, residuals, objective, jacobian = _evaluate(
        model, scaling, x_array, y_array, z, "initial"
    )
    derivative_checks = []
    if config.derivative_check:
        check = check_derivatives(
            model,
            scaling,
            z,
            x_array,
            config.finite_difference_step,
            config.derivative_rtol,
            config.derivative_atol,
            "initial",
        )
        derivative_checks.append(check)
        logger.event(
            "derivative_check",
            when="initial",
            passed=True,
            max_absolute_error=check.max_absolute_error,
            max_relative_error=check.max_relative_error,
            criterion=(
                f"abs(error) <= {config.derivative_atol} + "
                f"{config.derivative_rtol} * max(abs(AD), abs(FD))"
            ),
        )

    gradient = -jacobian.T @ residuals
    radius = config.trust_radius_initial
    history: list[dict[str, Any]] = []
    consecutive_invalid = 0
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration
        projected_gradient = _project_gradient(gradient, z, lower_z, upper_z)
        projected_gradient_norm = float(np.linalg.norm(projected_gradient))
        if projected_gradient_norm <= config.gradient_tol:
            return _finalize(
                Status.CONVERGED,
                "projected gradient satisfied tolerance",
                model,
                scaling,
                x_array,
                y_array,
                z,
                theta,
                prediction,
                residuals,
                objective,
                iteration - 1,
                radius,
                projected_gradient_norm,
                config,
                derivative_checks,
                history,
                logger,
            )

        hessian = jacobian.T @ jacobian
        tr_step = solve_boundary_trust_region(hessian, gradient, z, lower_z, upper_z, radius)
        if tr_step.predicted_reduction <= 0.0:
            return _finalize(
                Status.NONCONVERGENT,
                "trust-region subproblem produced no positive predicted reduction",
                model,
                scaling,
                x_array,
                y_array,
                z,
                theta,
                prediction,
                residuals,
                objective,
                iteration - 1,
                radius,
                projected_gradient_norm,
                config,
                derivative_checks,
                history,
                logger,
            )

        z_candidate = np.minimum(np.maximum(z + tr_step.step, lower_z), upper_z)
        try:
            new_theta, new_prediction, new_residuals, new_objective, new_jacobian = _evaluate(
                model, scaling, x_array, y_array, z_candidate, f"iteration {iteration}"
            )
        except SolverError:
            consecutive_invalid += 1
            radius = max(config.trust_radius_min, radius * config.shrink_factor)
            history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "failure_category": Status.ILLEGAL_FUNCTION_VALUE.value,
                    "actual_reduction": None,
                    "predicted_reduction": tr_step.predicted_reduction,
                    "reduction_ratio": None,
                    "trust_radius_before": radius / config.shrink_factor,
                    "trust_radius_after": radius,
                    "projected_gradient_norm": projected_gradient_norm,
                    "step_norm": float(np.linalg.norm(tr_step.step)),
                    "terminal_reason": tr_step.terminal_reason,
                }
            )
            logger.event(
                "iteration",
                iteration=iteration,
                accepted=False,
                failure_category=Status.ILLEGAL_FUNCTION_VALUE.value,
                predicted_reduction=tr_step.predicted_reduction,
                trust_radius_after=radius,
                decision="rejected because function produced non-finite values",
            )
            if radius <= config.trust_radius_min or consecutive_invalid >= 8:
                raise
            continue

        actual_reduction = objective - new_objective
        reduction_ratio = actual_reduction / tr_step.predicted_reduction
        accepted = reduction_ratio >= config.accept_ratio and actual_reduction > 0.0
        record = {
            "iteration": iteration,
            "accepted": accepted,
            "actual_reduction": float(actual_reduction),
            "predicted_reduction": float(tr_step.predicted_reduction),
            "reduction_ratio": float(reduction_ratio),
            "trust_radius_before": radius,
            "step_norm": float(np.linalg.norm(tr_step.step)),
            "projected_gradient_norm": projected_gradient_norm,
            "terminal_reason": tr_step.terminal_reason,
            "hit_trust_boundary": tr_step.hit_trust_boundary,
            "cg_iterations": tr_step.cg_iterations,
        }

        if accepted:
            consecutive_invalid = 0
            old_radius = radius
            z = z_candidate
            theta = new_theta
            prediction = new_prediction
            residuals = new_residuals
            objective = new_objective
            jacobian = new_jacobian
            gradient = -jacobian.T @ residuals
            if reduction_ratio < config.good_ratio:
                radius = max(config.trust_radius_min, radius * config.shrink_factor)
            elif reduction_ratio <= config.excellent_ratio:
                radius = min(config.trust_radius_max, radius / config.shrink_factor)
            elif reduction_ratio > config.excellent_ratio and tr_step.hit_trust_boundary:
                radius = min(config.trust_radius_max, radius * config.expand_factor)
            record["trust_radius_after"] = radius
            record["decision"] = "accepted; shrink/hold/expand based on agreement"
        else:
            old_radius = radius
            radius = max(config.trust_radius_min, radius * config.shrink_factor)
            record["trust_radius_after"] = radius
            record["decision"] = "rejected; actual reduction did not agree with model"

        history.append(record)
        logger.event(
            "iteration",
            iteration=iteration,
            accepted=accepted,
            actual_reduction=record["actual_reduction"],
            predicted_reduction=record["predicted_reduction"],
            reduction_ratio=record["reduction_ratio"],
            trust_radius_before=old_radius,
            trust_radius_after=radius,
            step_norm=record["step_norm"],
            projected_gradient_norm=projected_gradient_norm,
            cg_iterations=tr_step.cg_iterations,
            terminal_reason=tr_step.terminal_reason,
            decision=record["decision"],
        )

        if (
            accepted
            and float(np.linalg.norm(tr_step.step)) <= config.parameter_tol
            and radius <= config.trust_radius_min
        ):
            pg = float(np.linalg.norm(_project_gradient(gradient, z, lower_z, upper_z)))
            return _finalize(
                Status.NONCONVERGENT,
                "step size reached tolerance without projected-gradient stationarity",
                model,
                scaling,
                x_array,
                y_array,
                z,
                theta,
                prediction,
                residuals,
                objective,
                iteration,
                radius,
                pg,
                config,
                derivative_checks,
                history,
                logger,
            )

        if not accepted and radius <= config.trust_radius_min:
            pg = float(np.linalg.norm(_project_gradient(gradient, z, lower_z, upper_z)))
            return _finalize(
                Status.NONCONVERGENT,
                "minimum trust radius reached without acceptable reduction",
                model,
                scaling,
                x_array,
                y_array,
                z,
                theta,
                prediction,
                residuals,
                objective,
                iteration,
                radius,
                pg,
                config,
                derivative_checks,
                history,
                logger,
            )

    pg = float(np.linalg.norm(_project_gradient(gradient, z, lower_z, upper_z)))
    return _finalize(
        Status.NONCONVERGENT,
        "maximum iterations reached before projected-gradient tolerance",
        model,
        scaling,
        x_array,
        y_array,
        z,
        theta,
        prediction,
        residuals,
        objective,
        iterations,
        radius,
        pg,
        config,
        derivative_checks,
        history,
        logger,
    )


def _evaluate(
    model: CompiledModel,
    scaling: ParameterScaling,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    where: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    theta = scaling.to_external(z)
    prediction = model.evaluate(theta, x)
    finite_or_illegal(prediction, f"{where} prediction")
    residuals = y - prediction
    finite_or_illegal(residuals, f"{where} residual")
    jacobian, _ = normalized_jacobian_with_x(model, scaling, z, x)
    finite_or_illegal(jacobian, f"{where} jacobian")
    objective = 0.5 * float(residuals @ residuals)
    return theta, prediction, residuals, objective, jacobian


def _project_gradient(
    gradient: np.ndarray, z: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1.0e-10
) -> np.ndarray:
    projected = gradient.copy()
    at_lower = np.isfinite(lower) & (np.abs(z - lower) <= tol)
    at_upper = np.isfinite(upper) & (np.abs(z - upper) <= tol)
    projected[at_lower & (gradient >= 0.0)] = 0.0
    projected[at_upper & (gradient <= 0.0)] = 0.0
    return projected


def _finalize(
    status: Status,
    message: str,
    model: CompiledModel,
    scaling: ParameterScaling,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: np.ndarray,
    prediction: np.ndarray,
    residuals: np.ndarray,
    objective: float,
    iterations: int,
    radius: float,
    projected_gradient_norm: float,
    config: SolverConfig,
    derivative_checks: list,
    history: list[dict[str, Any]],
    logger: RunLogger,
) -> SolveResult:
    lower_z, upper_z = scaling.internal_bounds()
    final_jacobian, _ = normalized_jacobian_with_x(model, scaling, z, x)
    identity = identifiability(final_jacobian, config.rank_tolerance)
    final_status = status
    if not identity.identifiable:
        final_status = Status.UNIDENTIFIABLE
        message = f"rank-deficient Jacobian: {message}"

    if config.derivative_check:
        check = check_derivatives(
            model,
            scaling,
            z,
            x,
            config.finite_difference_step,
            config.derivative_rtol,
            config.derivative_atol,
            "final",
        )
        derivative_checks.append(check)
        logger.event(
            "derivative_check",
            when="final",
            passed=True,
            max_absolute_error=check.max_absolute_error,
            max_relative_error=check.max_relative_error,
            criterion=(
                f"abs(error) <= {config.derivative_atol} + "
                f"{config.derivative_rtol} * max(abs(AD), abs(FD))"
            ),
        )

    active: dict[str, str] = {}
    for name, value, lower, upper in zip(model.parameter_names, z, lower_z, upper_z):
        if np.isfinite(lower) and abs(float(value) - float(lower)) <= config.boundary_tolerance:
            active[name] = "lower"
        if np.isfinite(upper) and abs(float(value) - float(upper)) <= config.boundary_tolerance:
            active[name] = "upper"

    stationary = projected_gradient_norm <= config.gradient_tol
    stationarity_type = None
    if stationary:
        stationarity_type = "boundary_stationary" if active else "interior_stationary"

    check_dicts = [
        {
            "when": item.when,
            "passed": item.passed,
            "max_absolute_error": item.max_absolute_error,
            "max_relative_error": item.max_relative_error,
            "absolute_tolerance": item.tolerance_absolute,
            "relative_tolerance": item.tolerance_relative,
        }
        for item in derivative_checks
    ]
    diagnostics = {
        "objective_half_sum_squared_residuals": float(objective),
        "normalized_parameter_coordinates": [float(v) for v in z],
        "physical_parameter_scales": [float(v) for v in scaling.scales],
        "gradient_tolerance": config.gradient_tol,
        "rank_tolerance": config.rank_tolerance,
        "stationary_according_to_tolerance": stationary,
        "consecutive_rejection_policy": "shrink trust radius; report nonconvergent at minimum radius",
    }
    result = SolveResult(
        run_id=logger.run_id,
        status=final_status,
        message=message,
        expression=model.expression,
        parameter_names=model.parameter_names,
        parameters=[float(v) for v in theta],
        predicted=[float(v) for v in prediction],
        residuals=[float(v) for v in residuals],
        residual_sum_squares=float(residuals @ residuals),
        max_abs_residual=float(np.max(np.abs(residuals))),
        iterations=int(iterations),
        trust_radius=float(radius),
        projected_gradient_norm=float(projected_gradient_norm),
        stationarity_type=stationarity_type,
        active_bounds=active,
        identifiability={
            "numerical_rank": identity.numerical_rank,
            "parameter_count": identity.parameter_count,
            "singular_values": identity.singular_values,
            "condition_number": identity.condition_number,
            "identifiable": identity.identifiable,
            "parameter_correlation": identity.parameter_correlation,
        },
        derivative_checks=check_dicts,
        iteration_history=history,
        diagnostics=diagnostics,
    )
    logger.event(
        "run_finish",
        status=final_status.value,
        iterations=iterations,
        rss=result.residual_sum_squares,
        projected_gradient_norm=projected_gradient_norm,
        stationarity_type=stationarity_type,
        numerical_rank=identity.numerical_rank,
        parameter_count=identity.parameter_count,
        decision_basis=(
            "projected gradient tolerance, normalized Jacobian SVD rank, and final derivative review"
        ),
    )
    return result


def _validate_arrays(
    model: CompiledModel,
    x: np.ndarray,
    y: np.ndarray,
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
) -> None:
    if x.ndim != 1 or y.ndim != 1:
        raise SolverError(Status.INVALID_REQUEST, "x and y must be one-dimensional", http_status_code=400)
    if x.shape != y.shape or len(x) != model.n_points:
        raise SolverError(Status.INVALID_REQUEST, "x and y lengths must match the model data length", http_status_code=400)
    expected = model.n_parameters
    for name, array in (("initial", initial), ("lower", lower), ("upper", upper), ("scales", scales)):
        if array.shape != (expected,):
            raise SolverError(
                Status.INVALID_REQUEST,
                f"{name} must contain {expected} values",
                http_status_code=400,
            )
    if not np.all(np.isfinite(np.concatenate([x, y, initial, scales]))):
        raise SolverError(Status.INVALID_REQUEST, "x, y, initial values, and scales must be finite", http_status_code=400)
    for name, bound in (("lower", lower), ("upper", upper)):
        if not np.all(np.isfinite(bound) | np.isneginf(bound) | np.isposinf(bound)):
            raise SolverError(Status.INVALID_REQUEST, f"{name} must be finite or a signed infinity", http_status_code=400)
