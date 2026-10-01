import numpy as np
from scipy.optimize import linprog

from app.config import Tolerances
from app.core.types import FeasibilityResult, SolverEvent
from app.parser import ParsedProblem


def violation(x: np.ndarray, problem: ParsedProblem) -> float:
    ineq = float(np.max(problem.G @ x - problem.h)) if problem.n_inequalities else 0.0
    eq = float(np.max(np.abs(problem.A @ x - problem.b))) if problem.n_equalities else 0.0
    return max(0.0, ineq, eq)


def find_feasible_point(problem: ParsedProblem, tol: Tolerances) -> FeasibilityResult:
    events: list[SolverEvent] = []

    if problem.initial_point is not None:
        supplied_violation = violation(problem.initial_point, problem)
        events.append(
            SolverEvent(
                0,
                "phase_1",
                "validate_supplied_initial_point",
                {"max_primal_violation": supplied_violation, "primal_tol": tol.primal},
            )
        )
        if supplied_violation <= tol.primal:
            return FeasibilityResult(
                feasible=True,
                x=problem.initial_point.copy(),
                status="optimal",
                reason="Supplied initial point is feasible.",
                events=events,
                max_primal_violation=supplied_violation,
            )

    if problem.n_inequalities == 0 and problem.n_equalities == 0:
        x = problem.initial_point.copy() if problem.initial_point is not None else np.zeros(problem.n)
        events.append(SolverEvent(0, "phase_1", "unconstrained_initial_point", {"x": x.tolist()}))
        return FeasibilityResult(
            feasible=True,
            x=x,
            status="optimal",
            reason="No constraints; origin or supplied point is feasible.",
            events=events,
            max_primal_violation=0.0,
        )

    events.append(
        SolverEvent(
            0,
            "phase_1",
            "solve_feasibility_lp",
            {
                "reference_solver": "scipy.optimize.linprog(method='highs')",
                "objective": "zero objective; find a strictly feasible LP vertex",
                "n_variables": int(problem.n),
                "n_inequalities": int(problem.n_inequalities),
                "n_equalities": int(problem.n_equalities),
            },
        )
    )
    result = linprog(
        c=np.zeros(problem.n),
        A_ub=problem.G if problem.n_inequalities else None,
        b_ub=problem.h if problem.n_inequalities else None,
        A_eq=problem.A if problem.n_equalities else None,
        b_eq=problem.b if problem.n_equalities else None,
        bounds=[(None, None)] * problem.n,
        method="highs",
    )

    message = str(result.message)
    if result.status == 0:
        x = np.asarray(result.x, dtype=float)
        residual = violation(x, problem)
        events.append(
            SolverEvent(
                1,
                "phase_1",
                "feasibility_lp_solved",
                {"x": x.tolist(), "max_primal_violation": residual, "primal_tol": tol.primal},
            )
        )
        if residual <= tol.primal and np.all(np.isfinite(x)):
            return FeasibilityResult(
                feasible=True,
                x=x,
                status="optimal",
                reason="Independent HiGHS feasibility LP returned a feasible point.",
                events=events,
                max_primal_violation=residual,
                reference_message=message,
            )
        return FeasibilityResult(
            feasible=False,
            x=None,
            status="uncertain",
            reason="HiGHS reported success but independent residual validation failed.",
            events=events,
            max_primal_violation=residual,
            reference_message=message,
        )

    if result.status == 2:
        events.append(SolverEvent(1, "phase_1", "feasibility_lp_infeasible", {"message": message}))
        return FeasibilityResult(
            feasible=False,
            x=None,
            status="infeasible",
            reason="Independent HiGHS feasibility LP reported infeasibility.",
            events=events,
            max_primal_violation=None,
            reference_message=message,
        )

    reason_by_status = {
        1: "The feasibility LP is unbounded; a finite feasible point could not be obtained.",
        3: "The feasibility LP was not solved because a resource limit was reached.",
        4: "Numerical difficulties were reported by HiGHS.",
    }
    events.append(
        SolverEvent(
            1,
            "phase_1",
            "feasibility_lp_failed",
            {"linprog_status": int(result.status), "message": message},
        )
    )
    return FeasibilityResult(
        feasible=False,
        x=None,
        status="uncertain" if result.status in {3, 4} else "failed",
        reason=reason_by_status.get(result.status, f"linprog failed with status {result.status}."),
        events=events,
        reference_message=message,
    )
