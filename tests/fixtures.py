"""Reusable, locally constructed LP problems.

Hand-computed expected values are stored separately from the solver. SciPy is
used only by the randomized oracle test.
"""

from lp_service.models import Bound, Constraint, LPRequest


def two_variable_optimal() -> LPRequest:
    # min  x + y
    #     x + 2y >= 4
    #     x, y >= 0
    # Hand optimum: (0, 2), objective 2.
    return LPRequest(
        objective=[1.0, 1.0],
        sense="min",
        constraints=[Constraint(sense="ge", rhs=4.0, coefficients=[1.0, 2.0])],
        bounds=[Bound(lower=0.0), Bound(lower=0.0)],
    )


def degenerate_beale_cycling() -> LPRequest:
    # Beale's classical cycling example. Bland's rule terminates at objective
    # -0.395 with x1=1/2, x3=1 and all other variables zero.
    return LPRequest(
        objective=[-0.75, 150.0, -0.02, 6.0],
        sense="min",
        constraints=[
            Constraint(sense="le", rhs=0.0, coefficients=[0.25, -60.0, -0.125, 9.0]),
            Constraint(sense="le", rhs=0.0, coefficients=[0.5, -90.0, -0.25, 3.0]),
            Constraint(sense="le", rhs=1.0, coefficients=[0.0, 0.0, 1.0, 0.0]),
        ],
        bounds=[Bound(lower=0.0) for _ in range(4)],
    )


def redundant_constraints() -> LPRequest:
    # The first equality is duplicated; the third is their sum. All are
    # redundant relative to the first two but still give a unique vertex.
    return LPRequest(
        objective=[1.0, -1.0],
        sense="min",
        constraints=[
            Constraint(sense="eq", rhs=1.0, coefficients=[1.0, 0.0]),
            Constraint(sense="eq", rhs=2.0, coefficients=[0.0, 1.0]),
            Constraint(sense="eq", rhs=1.0, coefficients=[1.0, 0.0]),
            Constraint(sense="eq", rhs=3.0, coefficients=[1.0, 1.0]),
        ],
        bounds=[Bound(lower=0.0), Bound(lower=0.0)],
    )


def infeasible_with_certificate() -> LPRequest:
    return LPRequest(
        objective=[1.0, 1.0],
        sense="min",
        constraints=[
            Constraint(sense="le", rhs=-1.0, coefficients=[1.0, 0.0]),
            Constraint(sense="ge", rhs=0.0, coefficients=[1.0, 0.0]),
        ],
        bounds=[Bound(lower=0.0), Bound(lower=0.0)],
    )


def unbounded_with_free_upper() -> LPRequest:
    # min -x - y, 2x - y <= 4, x,y >= 0. Moving along (1,2) improves forever.
    return LPRequest(
        objective=[-1.0, -1.0],
        sense="min",
        constraints=[Constraint(sense="le", rhs=4.0, coefficients=[2.0, -1.0])],
        bounds=[Bound(lower=0.0), Bound(lower=0.0)],
    )


def bounded_and_fixed() -> LPRequest:
    # x fixed at 1 and 0 <= y <= 2. min x-y gives y=2 and objective -1.
    return LPRequest(
        objective=[1.0, -1.0],
        sense="min",
        constraints=[Constraint(sense="eq", rhs=3.0, coefficients=[1.0, 1.0])],
        bounds=[Bound(lower=1.0, upper=1.0), Bound(lower=0.0, upper=2.0)],
    )
