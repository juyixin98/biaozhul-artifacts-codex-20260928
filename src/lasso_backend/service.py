"""Application service that composes parsing, preprocessing, and solving."""

from __future__ import annotations

import numpy as np

from .config import Settings
from .diagnostics import RequestContext, build_result_diagnostic
from .errors import LassoRequestError
from .models import LassoPathRequest, LassoPathResponse, LambdaPointResponse
from .parsing import parse_request
from .preprocessing import PreprocessingError, prepare_data
from .solver import LassoPathResult, SolverError, lasso_path


def solve_request(
    request: LassoPathRequest,
    settings: Settings,
    context: RequestContext,
) -> tuple[LassoPathResult, object]:
    x, y = parse_request(request, settings)
    context.add_event(
        "parse",
        "accepted",
        "request payload converted to numeric arrays",
        n_rows=x.shape[0],
        n_features=x.shape[1],
    )

    prepared = prepare_data(
        x,
        y,
        zero_variance_policy=request.zero_variance_policy,
        variance_tolerance=request.variance_tolerance,
    )
    context.add_event(
        "preprocess",
        "accepted",
        "frozen feature centering and scaling completed",
        sparse=prepared.is_sparse,
        constant_columns=int(np_count_zero_variance(prepared)),
    )

    result = lasso_path(
        prepared,
        lambdas=request.lambdas,
        n_alphas=request.n_alphas,
        lambda_min_ratio=request.lambda_min_ratio,
        lambda_order=request.lambda_order,
        warm_start=request.warm_start,
        max_iter=request.max_iter,
        tol=request.tol,
    )
    failed = sum(not item.converged for item in result.diagnostics)
    context.add_event(
        "solve",
        "accepted" if result.converged else "undetermined",
        (
            "all lambda points accepted by KKT criterion"
            if result.converged
            else f"{failed} lambda point(s) could not be accepted"
        ),
        n_lambdas=len(result.diagnostics),
        failed_points=failed,
        warm_start=result.warm_start,
    )
    return result, prepared


def np_count_zero_variance(prepared) -> int:
    return int(prepared.zero_variance.sum())


def to_response(
    request: LassoPathRequest,  # retained for future endpoint-level options
    prepared,
    result: LassoPathResult,
    context: RequestContext,
) -> LassoPathResponse:
    diagnostics = build_result_diagnostic(context, prepared, result)
    points = []
    for index, item in enumerate(result.diagnostics):
        points.append(
            LambdaPointResponse(
                lambda_value=item.lambda_value,
                coefficients=result.coefficients[index].tolist(),
                intercept=float(result.intercepts[index]),
                support=np.flatnonzero(result.support[index]).tolist(),
                iterations=item.iterations,
                converged=item.converged,
                status=item.status,
                reason=item.reason,
                kkt_violation=item.kkt_violation,
                duality_gap=item.duality_gap,
                primal_objective=item.primal_objective,
            )
        )

    return LassoPathResponse(
        request_id=context.request_id,
        accepted=result.converged,
        decision="accepted" if result.converged else "undetermined",
        reason=diagnostics["reason"],
        lambdas=result.lambdas.tolist(),
        points=points,
        diagnostics=diagnostics,
    )


def run_lasso(
    request: LassoPathRequest,
    settings: Settings,
    http_request,
) -> LassoPathResponse:
    context = RequestContext()
    http_request.state.request_id = context.request_id
    try:
        result, prepared = solve_request(request, settings, context)
        return to_response(request, prepared, result, context)
    except (LassoRequestError, PreprocessingError, SolverError) as exc:
        context.add_event(
            "reject",
            "rejected",
            str(exc),
            error_category=type(exc).__name__,
        )
        raise
