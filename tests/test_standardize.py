import numpy as np

from lp_service.parser import parse_request
from lp_service.standardize import standardize

from lp_service.models import Bound, Constraint, LPRequest


def test_upper_only_variable_is_nonnegative_transform_without_extra_upper_row():
    request = LPRequest(
        objective=[-1.0],
        constraints=[Constraint(sense="le", rhs=1.0, coefficients=[1.0])],
        bounds=[Bound(lower=None, upper=2.0)],
    )
    model = standardize(parse_request(request))
    assert [meta.kind for meta in model.columns if meta.variable == 0] == ["upper_only"]
    assert all(meta.kind == "constraint" for meta in model.rows)
    # x=2-z: x<=1 becomes -z <= -1, represented after RHS sign normalization.
    assert model.d[0] == pytest.approx(1.0)
    assert model.A[0, 0] == pytest.approx(1.0)


def test_fixed_variable_shifts_rhs_and_objective_constant():
    request = LPRequest(
        objective=[2.0, -1.0],
        constraints=[Constraint(sense="eq", rhs=5.0, coefficients=[1.0, 1.0])],
        bounds=[Bound(lower=2.0, upper=2.0), Bound(lower=0.0)],
    )
    model = standardize(parse_request(request))
    assert model.d[0] == pytest.approx(3.0)
    assert model.objective_constant == pytest.approx(4.0)
    assert model.c[0] == pytest.approx(-1.0)


import pytest
