import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from app.config import Tolerances
from app.models import LinearConstraint, SolveRequest
from app.parser import parse_qp_request
from app.service import solve_qp


def constraint(coeffs, op, rhs, name=None):
    return LinearConstraint(name=name, coefficients=[float(v) for v in coeffs], op=op, rhs=float(rhs))


def boundary_request(request_id="boundary"):
    return SolveRequest(
        request_id=request_id,
        objective={"quadratic": [[2.0, 0.0], [0.0, 2.0]], "linear": [-2.0, -4.0]},
        constraints=[
            constraint([1, 1], "<=", 1, "sum_leq_1"),
            constraint([1, 0], ">=", 0, "x_nonnegative"),
            constraint([0, 1], ">=", 0, "y_nonnegative"),
        ],
    )


def redundant_constraints_request():
    return SolveRequest(
        request_id="redundant-duplicates",
        objective={"quadratic": [[2.0, 0.0], [0.0, 2.0]], "linear": [-2.0, -4.0]},
        constraints=[
            constraint([1, 1], "<=", 1, "binding_sum"),
            constraint([2, 2], "<=", 2, "duplicate_sum_scaled"),
            constraint([-1, -1], ">=", -1, "duplicate_sum_ge_form"),
            constraint([1, 0], ">=", 0, "x_nonnegative"),
            constraint([0, 1], ">=", 0, "y_nonnegative"),
            constraint([1, 1], "<=", 3, "strictly_redundant_sum"),
        ],
    )


def near_degenerate_request():
    epsilon = 1e-5
    return SolveRequest(
        request_id="near-degenerate",
        objective={
            "quadratic": [
                [1.0 + epsilon, epsilon - 1.0],
                [epsilon - 1.0, 1.0 + epsilon],
            ],
            "linear": [-0.5, -1.5],
        },
        constraints=[
            constraint([1, 1], "<=", 1, "weak_sum_bound"),
            constraint([1, 1 + 2 * epsilon], "<=", 1, "near_duplicate_bound"),
            constraint([1, 0], ">=", 0, "x_nonnegative"),
            constraint([0, 1], ">=", 0, "y_nonnegative"),
        ],
        tolerances={"rank_tol": 1e-12, "active_tol": 1e-9},
    )


def negative_multiplier_request():
    return SolveRequest(
        request_id="negative-multiplier-drop",
        objective={"quadratic": [[2.0, 0.0], [0.0, 2.0]], "linear": [0.0, 0.0]},
        constraints=[
            constraint([1, 0], ">=", 0, "x_nonnegative"),
            constraint([1, 0], "<=", 1, "x_upper"),
            constraint([0, 1], ">=", 0, "y_nonnegative"),
            constraint([0, 1], "<=", 1, "y_upper"),
        ],
        initial_point=[1.0, 1.0],
    )


def rank_deficient_equalities_request():
    return SolveRequest(
        request_id="rank-deficient-equalities",
        objective={"quadratic": [[2.0, 0.0], [0.0, 2.0]], "linear": [-2.0, -2.0]},
        constraints=[
            constraint([1, 1], "=", 1, "equality_original"),
            constraint([2, 2], "=", 2, "equality_scaled_copy"),
            constraint([1, 0], ">=", 0, "x_nonnegative"),
            constraint([0, 1], ">=", 0, "y_nonnegative"),
        ],
    )


def infeasible_request():
    return SolveRequest(
        request_id="infeasible-box",
        objective={"quadratic": [[2.0, 0.0], [0.0, 2.0]], "linear": [0.0, 0.0]},
        constraints=[
            constraint([1, 0], "<=", 0, "x_le_0"),
            constraint([1, 0], ">=", 1, "x_ge_1"),
        ],
    )


def scipy_slsqp_reference(request: SolveRequest):
    parsed = parse_qp_request(request, Tolerances())

    def objective(x):
        return float(0.5 * x @ parsed.H @ x + parsed.c @ x)

    def gradient(x):
        return parsed.H @ x + parsed.c

    result = minimize(
        objective,
        x0=np.zeros(parsed.n),
        jac=gradient,
        method="SLSQP",
        constraints=(
            [{"type": "ineq", "fun": lambda x, i=i: parsed.h[i] - parsed.G[i] @ x, "jac": lambda x, i=i: -parsed.G[i]}
             for i in range(parsed.n_inequalities)]
            + [{"type": "eq", "fun": lambda x, i=i: parsed.b[i] - parsed.A[i] @ x, "jac": lambda x, i=i: -parsed.A[i]}
               for i in range(parsed.n_equalities)]
        ),
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert result.success, result.message
    return result


def scipy_highs_feasibility_reference(request: SolveRequest):
    parsed = parse_qp_request(request, Tolerances())
    return linprog(
        c=np.zeros(parsed.n),
        A_ub=parsed.G if parsed.n_inequalities else None,
        b_ub=parsed.h if parsed.n_inequalities else None,
        A_eq=parsed.A if parsed.n_equalities else None,
        b_eq=parsed.b if parsed.n_equalities else None,
        bounds=[(None, None)] * parsed.n,
        method="highs",
    )


@pytest.fixture
def solve():
    return solve_qp
