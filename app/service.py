from __future__ import annotations

import logging
import uuid

import numpy as np
import scipy

from app import __version__
from dataclasses import asdict
from app.core.diagnostics import FitError, FitResult, FitStatus
from app.core.expression import ExpressionModel, RestrictedExpression, check_jacobian
from app.core.solver import solve_least_squares
from app.logging_config import fingerprint_payload, log_event, run_id_var
from app.schemas import (
    DerivativeCheckResponse,
    FitRequest,
    FitResponse,
    IdentifiabilityResponse,
    IterationResponse,
)

logger = logging.getLogger("nls.service")


def _finite_array(values: list[float | None] | None, name: str) -> np.ndarray:
    if values is None:
        raise FitError(FitStatus.INVALID_INPUT, f"{name} is required")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise FitError(FitStatus.INVALID_INPUT, f"{name} must contain only finite numbers")
    return array


def validate_and_prepare(request: FitRequest):
    n = len(request.parameter_names)
    if not request.x:
        raise FitError(FitStatus.INVALID_INPUT, "x must contain at least one observation")
    if len(request.x) != len(request.y):
        raise FitError(FitStatus.INVALID_INPUT, "x and y must have equal length")
    if len(request.initial) != n:
        raise FitError(FitStatus.INVALID_INPUT, "initial length must match parameter_names")
    if len(request.bounds.lower) != n or len(request.bounds.upper) != n:
        raise FitError(FitStatus.INVALID_INPUT, "bounds length must match parameter_names")
    if request.variable_name in request.parameter_names:
        raise FitError(FitStatus.INVALID_INPUT, "variable_name cannot also be a parameter")

    x = _finite_array(request.x, "x")
    y = _finite_array(request.y, "y")
    initial = _finite_array(request.initial, "initial")
    scales = None
    if request.scales is not None:
        if len(request.scales) != n:
            raise FitError(FitStatus.INVALID_INPUT, "scales length must match parameter_names")
        scales = _finite_array(request.scales, "scales")
        if np.any(scales <= 0.0):
            raise FitError(FitStatus.INVALID_INPUT, "scales must be strictly positive")
    if request.weights is not None:
        if len(request.weights) != len(y):
            raise FitError(FitStatus.INVALID_INPUT, "weights and y must have equal length")
        weights = _finite_array(request.weights, "weights")
        if np.any(weights <= 0.0):
            raise FitError(FitStatus.INVALID_INPUT, "weights must be strictly positive")
    else:
        weights = None

    lower = np.array(
        [-np.inf if value is None else float(value) for value in request.bounds.lower],
        dtype=float,
    )
    upper = np.array(
        [np.inf if value is None else float(value) for value in request.bounds.upper],
        dtype=float,
    )
    if not np.all(np.isfinite(lower) | np.isneginf(lower)) or not np.all(
        np.isfinite(upper) | np.isposinf(upper)
    ):
        raise FitError(FitStatus.INVALID_INPUT, "bounds must be finite, null, or infinite")
    if np.any(lower >= upper):
        raise FitError(FitStatus.INVALID_INPUT, "each lower bound must be below upper bound")
    if np.any(initial < lower) or np.any(initial > upper):
        raise FitError(FitStatus.INVALID_INPUT, "initial point must be inside bounds")

    parsed = RestrictedExpression(
        request.expression,
        request.parameter_names,
        variable_name=request.variable_name,
    )
    model = ExpressionModel(parsed, x, y, weights)
    return model, initial, lower, upper, scales


def build_response(
    result: FitResult,
    run_id: str,
    input_fingerprint: str,
) -> FitResponse:
    active_lower, active_upper = result.active_map()
    return FitResponse(
        run_id=run_id,
        input_fingerprint=input_fingerprint,
        success=result.success,
        status=result.status,
        message=result.message,
        reason=result.reason,
        parameters=result.parameter_map(),
        residuals=[float(v) for v in result.residuals],
        cost=result.cost,
        projected_gradient_norm=result.projected_gradient_norm,
        stationarity_type=result.stationarity_type,
        active_lower=active_lower,
        active_upper=active_upper,
        scales={
            name: float(value)
            for name, value in zip(result.parameter_names, result.scales)
        },
        nfev=result.nfev,
        njev=result.njev,
        invalid_evaluations=result.invalid_evaluations,
        final_radius=result.final_radius,
        iterations=[IterationResponse.from_core(item) for item in result.iterations],
        identifiability=IdentifiabilityResponse.from_core(result.identifiability),
        derivative_check=(
            DerivativeCheckResponse(**asdict(result.derivative_check))
            if result.derivative_check is not None
            else None
        ),
    )


def fit(request: FitRequest) -> tuple[FitResponse, FitResult]:
    run_id = str(uuid.uuid4())
    token = run_id_var.set(run_id)
    input_fingerprint = fingerprint_payload(request.model_dump(mode="json"))
    try:
        log_event(
            logger,
            "fit_started",
            run_id=run_id,
            input_fingerprint=input_fingerprint,
            service_version=__version__,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            observations=len(request.x),
            parameters=len(request.parameter_names),
            step="validate_parse",
        )
        model, initial, lower, upper, scales = validate_and_prepare(request)
        log_event(
            logger,
            "model_parsed",
            run_id=run_id,
            input_fingerprint=input_fingerprint,
            expression=request.expression,
            parameter_names=request.parameter_names,
            step="parse_complete",
            criterion="restricted AST accepted",
        )

        derivative_report = None
        if request.options.check_derivatives:
            residual_at_initial, analytical_jacobian = model.residual_and_jacobian(initial)
            derivative_report = check_jacobian(
                model.residual,
                analytical_jacobian,
                initial,
                (lower, upper),
                request.parameter_names,
                tolerance=request.options.derivative_tolerance,
                relative_step=request.options.finite_difference_step,
            )
            log_event(
                logger,
                "derivative_check_complete",
                run_id=run_id,
                input_fingerprint=input_fingerprint,
                passed=derivative_report.passed,
                max_relative_error=derivative_report.max_relative_error,
                worst_parameter=derivative_report.worst_parameter,
                worst_observation=derivative_report.worst_observation,
                step="finite_difference_audit",
                criterion=(
                    f"max_relative_error <= {request.options.derivative_tolerance}"
                ),
            )
            if not derivative_report.passed:
                raise FitError(
                    FitStatus.DERIVATIVE_MISMATCH,
                    (
                        "automatic derivative failed finite-difference review: "
                        f"max relative error {derivative_report.max_relative_error:.6g}"
                    ),
                )

        result = solve_least_squares(
            model.residual,
            model.residual_and_jacobian,
            initial,
            (lower, upper),
            scales=scales,
            max_iterations=request.options.max_iterations,
            radius=request.options.initial_radius,
            min_radius=request.options.min_radius,
            gtol=request.options.gtol,
            xtol=request.options.xtol,
            ftol=request.options.ftol,
            max_invalid_trials=request.options.max_invalid_trials,
            derivative_check=derivative_report,
            parameter_names=request.parameter_names,
        )
        for item in result.iterations:
            log_event(
                logger,
                "iteration",
                run_id=run_id,
                input_fingerprint=input_fingerprint,
                service_version=__version__,
                numpy_version=np.__version__,
                scipy_version=scipy.__version__,
                iteration=item.iteration,
                progress=f"{item.iteration}/{request.options.max_iterations}",
                accepted=item.accepted,
                cost=item.cost,
                actual_reduction=item.actual_reduction,
                predicted_reduction=item.predicted_reduction,
                reduction_ratio=item.reduction_ratio,
                radius=item.radius,
                step_norm_scaled=item.step_norm_scaled,
                projected_gradient_norm=item.projected_gradient_norm,
                active_lower=item.active_lower,
                active_upper=item.active_upper,
                bound_hit=item.bound_hit,
                step="trust_region_iteration",
                criterion=item.judgment,
            )

        ident = result.identifiability
        log_event(
            logger,
            "fit_finished",
            run_id=run_id,
            input_fingerprint=input_fingerprint,
            success=result.success,
            status=result.status.value,
            status_value=result.status.value,
            message=result.message,
            reason=result.reason,
            cost=result.cost,
            projected_gradient_norm=result.projected_gradient_norm,
            rank=ident.rank if ident else None,
            parameter_count=ident.parameter_count if ident else None,
            iterations_run=len(result.iterations),
            step="finalize",
            criterion=f"{result.status.value}: {result.reason}",
        )
        response = build_response(result, run_id, input_fingerprint)
        return response, result
    except FitError as exc:
        exc.run_id = run_id
        exc.input_fingerprint = input_fingerprint
        log_event(
            logger,
            "fit_failed",
            level=logging.WARNING,
            run_id=run_id,
            input_fingerprint=input_fingerprint,
            status=exc.status.value,
            message=exc.message,
            reason=str(exc),
            step="failure",
            criterion="distinct failure category returned without success marking",
        )
        raise
    except Exception:
        log_event(
            logger,
            "fit_failed",
            level=logging.ERROR,
            run_id=run_id,
            input_fingerprint=input_fingerprint,
            status=FitStatus.INTERNAL_ERROR.value,
            step="failure",
            criterion="unknown exception must not be reported as success",
        )
        raise
    finally:
        run_id_var.reset(token)
