import numpy as np
import pytest
from scipy.optimize import linprog

from lp_service.models import Bound, Constraint, LPRequest
from lp_service.solver import solve_lp


def _scipy_min(request: LPRequest):
    c = np.asarray(request.objective)
    if request.sense == "max":
        c = -c
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for row in request.constraints:
        if row.sense == "le":
            A_ub.append(row.coefficients)
            b_ub.append(row.rhs)
        elif row.sense == "ge":
            A_ub.append(np.asarray(row.coefficients) * -1.0)
            b_ub.append(-row.rhs)
        else:
            A_eq.append(row.coefficients)
            b_eq.append(row.rhs)
    bounds = []
    for bound in request.bounds:
        if bound is None:
            bounds.append((0.0, None))
        else:
            bounds.append((bound.lower, bound.upper))
    return linprog(
        c,
        A_ub=np.asarray(A_ub) if A_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=np.asarray(A_eq) if A_eq else None,
        b_eq=np.asarray(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )


@pytest.mark.parametrize("seed", [7, 19, 42])
def test_random_feasible_bounded_objectives_match_scipy(seed: int):
    rng = np.random.default_rng(seed)
    n, m = 4, 6
    x = rng.uniform(0.0, 2.0, n)
    A = rng.normal(size=(m, n))
    slack = rng.uniform(0.0, 0.2, m)
    b = A @ x + slack
    c = rng.normal(size=n)
    request = LPRequest(
        objective=c.tolist(),
        sense="min",
        constraints=[
            Constraint(sense="le", rhs=float(value), coefficients=row.tolist())
            for row, value in zip(A, b)
        ],
        bounds=[Bound(lower=0.0, upper=4.0) for _ in range(n)],
    )

    oracle = _scipy_min(request)
    assert oracle.success
    result = solve_lp(request, request_id=f"oracle-{seed}").response
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(float(oracle.fun), rel=1e-7, abs=1e-7)
    assert result.diagnostic.accepted is True


def test_maximization_objective_matches_scipy_sign_convention():
    request = LPRequest(
        objective=[2.0, 3.0],
        sense="max",
        constraints=[Constraint(sense="le", rhs=8.0, coefficients=[1.0, 2.0])],
        bounds=[Bound(lower=0.0, upper=5.0), Bound(lower=0.0, upper=5.0)],
    )
    oracle = _scipy_min(request)
    result = solve_lp(request, request_id="max-scipy").response
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(-float(oracle.fun))
    assert result.diagnostic.checks["weak_duality_gap_zero"] is True


def test_scipy_agrees_on_infeasible_and_unbounded_failure_categories():
    infeasible = LPRequest(
        objective=[1.0],
        constraints=[
            Constraint(sense="le", rhs=-1.0, coefficients=[1.0]),
        ],
        bounds=[Bound(lower=0.0)],
    )
    assert _scipy_min(infeasible).status == 2
    assert solve_lp(infeasible, request_id="oracle-infeasible").response.status == "infeasible"

    unbounded = LPRequest(
        objective=[-1.0],
        constraints=[],
        bounds=[Bound(lower=0.0)],
    )
    assert _scipy_min(unbounded).status == 3
    assert solve_lp(unbounded, request_id="oracle-unbounded").response.status == "unbounded"
