"""Application service combining parsing, simplex, mapping and diagnostics."""

from dataclasses import dataclass

import numpy as np

from .config import SolverConfig
from .evidence import EvidenceReport, primal_residuals, verify_result
from .kernel import two_phase_simplex
from .mapping import map_result
from .models import LPFResponse
from .parser import parse_request
from .standardize import standardize


@dataclass
class ServiceResult:
    response: LPFResponse


def solve_lp(request, request_id: str, config: SolverConfig = SolverConfig()) -> ServiceResult:
    problem = parse_request(request)
    model = standardize(problem)
    kernel = two_phase_simplex(model, config=config, request_id=request_id)
    mapped = map_result(problem, model, kernel)
    evidence = verify_result(problem, mapped, config)

    residual = bound = None
    if mapped.solution is not None:
        residual, bound = primal_residuals(problem, mapped.solution)

    response = LPFResponse(
        status=mapped.status if evidence.accepted or mapped.status == "numerical_uncertain" else "numerical_uncertain",
        request_id=request_id,
        objective_value=mapped.objective_value,
        solution=None if mapped.solution is None else mapped.solution.tolist(),
        residuals=None if residual is None else residual.tolist(),
        bound_violations=None if bound is None else bound.tolist(),
        duals=None
        if mapped.constraint_duals is None
        else {
            "constraints": mapped.constraint_duals.tolist(),
            "lower_bounds": mapped.lower_duals.tolist(),
            "upper_bounds": mapped.upper_duals.tolist(),
            "fixed_variables": mapped.fixed_duals.tolist(),
        },
        unbounded_direction=None if mapped.direction is None else mapped.direction.tolist(),
        certificate=None
        if mapped.constraint_certificate is None
        else {
            "constraints": mapped.constraint_certificate.tolist(),
            "lower_bounds": mapped.lower_certificate.tolist(),
            "upper_bounds": mapped.upper_certificate.tolist(),
            "contradiction": mapped.certificate_contradiction,
        },
        diagnostic={
            "accepted": evidence.accepted,
            "checks": evidence.checks,
            "residuals": evidence.residuals,
            "message": evidence.message,
        },
    )
    return ServiceResult(response=response)
