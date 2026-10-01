"""Run synthetic cases against SciPy SLSQP/HiGHS and print solver evidence."""

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import (  # noqa: E402
    boundary_request,
    infeasible_request,
    near_degenerate_request,
    rank_deficient_equalities_request,
    redundant_constraints_request,
    scipy_highs_feasibility_reference,
    scipy_slsqp_reference,
)
from app.service import solve_qp  # noqa: E402


logging.disable(logging.CRITICAL)


FEASIBLE_CASES = [
    ("boundary optimum", boundary_request, "scipy_slsqp"),
    ("redundant constraints", redundant_constraints_request, "scipy_slsqp"),
    ("near-degenerate", near_degenerate_request, "scipy_slsqp"),
    ("rank-deficient equalities", rank_deficient_equalities_request, "analytic_[0.5,0.5]"),
]


def main() -> int:
    rows = []
    for label, factory, reference_name in FEASIBLE_CASES:
        request = factory()
        response = solve_qp(request)
        x = np.asarray(response["solution"]["x"]) if response["solution"] else None
        if reference_name == "scipy_slsqp":
            reference_x = scipy_slsqp_reference(request).x
            x_delta = float(np.max(np.abs(x - reference_x)))
        else:
            reference_x = np.asarray([0.5, 0.5])
            x_delta = float(np.max(np.abs(x - reference_x)))
        rows.append(
            {
                "case": label,
                "status": response["status"],
                "x": x.tolist() if x is not None else None,
                "reference": reference_name,
                "reference_x": reference_x.tolist(),
                "x_delta_inf": x_delta,
                "active_inequality_indices": response["active_set"]["solver_active_inequality_indices"],
                "equality_basis_indices": response["active_set"]["equality_basis_indices"],
                "dependent_equality_indices": response["active_set"]["dependent_equality_indices"],
                "events": [event["type"] for event in response["events"]],
                "kkt_residuals": {
                    key: response["kkt_residuals"][key]
                    for key in (
                        "stationarity_inf",
                        "primal_inequality_violation",
                        "primal_equality_violation",
                        "dual_feasibility_violation",
                        "complementarity_max",
                    )
                },
            }
        )

    infeasible_case = infeasible_request()
    infeasible = solve_qp(infeasible_case)
    highs = scipy_highs_feasibility_reference(infeasible_case)
    rows.append(
        {
            "case": "infeasible box",
            "status": infeasible["status"],
            "highs_status": int(highs.status),
            "failure_category": infeasible["failure_category"],
            "reason": infeasible["reason"],
            "events": [event["type"] for event in infeasible["events"]],
        }
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0 if all(row["status"] in {"optimal", "infeasible"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
