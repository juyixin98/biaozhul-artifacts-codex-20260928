import hashlib
import json
import os
import platform
from importlib.metadata import version
from typing import Any

import numpy as np

from app.config import SERVICE_NAME, SERVICE_VERSION, SOLVER_NAME, Tolerances
from app.core.active_set import solve_active_set
from app.core.diagnostics import diagnose
from app.core.feasibility import find_feasible_point
from app.errors import QPError
from app.models import FeasibilityRequest, SolveRequest
from app.observability import configure_logging, log_event, new_request_id, request_id_var
from app.parser import parse_feasibility_request, parse_qp_request


logger = configure_logging()


def runtime_metadata() -> dict[str, str]:
    return {
        "service": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "solver": SOLVER_NAME,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "processing_location": f"{platform.node()}:{os.getpid()}",
    }


def request_fingerprint(request: SolveRequest | FeasibilityRequest) -> str:
    encoded = json.dumps(request.model_dump(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_events(events) -> list[dict[str, Any]]:
    return [event.to_dict() for event in events]


def solve_qp(request: SolveRequest) -> dict[str, Any]:
    request_id = new_request_id(request.request_id)
    token = request_id_var.set(request_id)
    fingerprint = request_fingerprint(request)
    try:
        log_event(
            logger,
            20,
            "QP solve request accepted",
            request_fingerprint=fingerprint,
            n_constraints=len(request.constraints),
        )
        tol = Tolerances.from_api(request.tolerances.model_dump(exclude_none=True) if request.tolerances else None)
        problem = parse_qp_request(request, tol)
        result = solve_active_set(problem, tol)

        diagnostics = None
        final_status = result.status
        failure_reasons: list[str] = []
        uncertainties: list[str] = []
        notes: list[str] = []

        if result.status == "optimal" and np.all(np.isfinite(result.x)):
            diagnostics = diagnose(problem, result.x, result, tol)
            failure_reasons.extend(diagnostics.failure_reasons)
            uncertainties = diagnostics.uncertainties
            notes = [
                text
                for text in result.warnings
                if "dependent equalities represented by basis rows" in text
            ]
            if failure_reasons:
                final_status = "failed"
                result.reason = "Final KKT validation rejected the active-set result."
            elif uncertainties:
                final_status = "uncertain"
                result.reason = "Solver completed, but numerical diagnostics are uncertain."

        response = {
            "request_id": request_id,
            "request_fingerprint_sha256": fingerprint,
            "status": final_status,
            "failure_category": None
            if final_status == "optimal"
            else ("infeasible" if final_status == "infeasible" else "numerical_validation"),
            "reason": result.reason,
            "failure_reasons": failure_reasons,
            "uncertainties": uncertainties,
            "notes": notes,
            "solution": None
            if not np.all(np.isfinite(result.x))
            else {
                "x": result.x.tolist(),
                "variable_names": problem.variable_names,
            },
            "active_set": {
                "solver_active_inequality_indices": result.active_inequality_indices,
                "solver_active_inequality_names": result.active_inequality_names,
                "equality_basis_indices": result.equality_basis_indices,
                "dependent_equality_indices": result.dependent_equality_indices,
            },
            "kkt_residuals": diagnostics.to_dict() if diagnostics else None,
            "iteration_count": result.iterations,
            "events": _public_events(result.events),
            "tolerances": {
                "primal": tol.primal,
                "dual": tol.dual,
                "complementarity": tol.complementarity,
                "active": tol.active,
                "rank": tol.rank,
                "step": tol.step,
            },
            "runtime": runtime_metadata(),
        }
        log_event(
            logger,
            30 if final_status != "optimal" else 20,
            "QP solve request completed",
            status=final_status,
            iterations=result.iterations,
            active_indices=result.active_inequality_indices,
            failure_reasons=failure_reasons,
            uncertainties=uncertainties,
        )
        return response
    except QPError as exc:
        exc.request_id = request_id
        raise
    finally:
        request_id_var.reset(token)


def find_feasibility(request: FeasibilityRequest) -> dict[str, Any]:
    request_id = new_request_id(request.request_id)
    token = request_id_var.set(request_id)
    fingerprint = request_fingerprint(request)
    try:
        log_event(logger, 20, "Feasibility request accepted", request_fingerprint=fingerprint)
        tol = Tolerances.from_api(request.tolerances.model_dump(exclude_none=True) if request.tolerances else None)
        n, problem = parse_feasibility_request(request, tol)
        result = find_feasible_point(problem, tol)
        response = {
            "request_id": request_id,
            "request_fingerprint_sha256": fingerprint,
            "status": "feasible" if result.feasible else result.status,
            "failure_category": None if result.feasible else result.status,
            "reason": result.reason,
            "failure_reasons": [] if result.feasible else [result.reason],
            "uncertainties": [],
            "notes": [],
            "n": n,
            "solution": None if result.x is None else {"x": result.x.tolist(), "variable_names": problem.variable_names},
            "max_primal_violation": result.max_primal_violation,
            "events": _public_events(result.events),
            "reference_solver": result.reference_solver,
            "reference_message": result.reference_message,
            "runtime": runtime_metadata(),
        }
        log_event(
            logger,
            30 if not result.feasible else 20,
            "Feasibility request completed",
            status=response["status"],
            reason=result.reason,
        )
        return response
    except QPError as exc:
        exc.request_id = request_id
        raise
    finally:
        request_id_var.reset(token)
