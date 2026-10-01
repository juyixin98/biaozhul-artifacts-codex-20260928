"""Application service layer: request compilation and result protocol."""

from __future__ import annotations

import numpy as np

from app.core import solve_least_squares
from app.observability import RunLogger
from app.parsing import CompiledModel
from app.schemas import SolveRequest
from app.status import SolverError, Status


def solve_request(request: SolveRequest, run_id: str, logger: RunLogger) -> dict:
    x = np.asarray(request.x, dtype=float)
    y = np.asarray(request.y, dtype=float)
    if x.shape != y.shape:
        raise SolverError(Status.INVALID_REQUEST, "x and y must have equal lengths", http_status_code=400)

    parameter_names = [parameter.name for parameter in request.parameters]
    initial = np.asarray([parameter.initial for parameter in request.parameters], dtype=float)
    lower = np.asarray([parameter.lower for parameter in request.parameters], dtype=float)
    upper = np.asarray([parameter.upper for parameter in request.parameters], dtype=float)
    scales = np.asarray([parameter.scale for parameter in request.parameters], dtype=float)

    model = CompiledModel(request.expression, parameter_names, len(x))
    result = solve_least_squares(
        model=model,
        x=x,
        y=y,
        initial=initial,
        lower=lower,
        upper=upper,
        scales=scales,
        config=request.config,
        logger=logger,
    )

    if result.status != Status.CONVERGED:
        payload = result.to_dict()
        raise SolverError(
            result.status,
            result.message,
            details={
                "residual_sum_squares": payload["residual_sum_squares"],
                "projected_gradient_norm": payload["projected_gradient_norm"],
                "iterations": payload["iterations"],
                "stationarity_type": payload["stationarity_type"],
                "identifiability": payload["identifiability"],
                "active_bounds": payload["active_bounds"],
            },
            http_status_code=422,
        )

    payload = result.to_dict()
    payload["log_tail"] = logger.events[-20:]
    return payload
