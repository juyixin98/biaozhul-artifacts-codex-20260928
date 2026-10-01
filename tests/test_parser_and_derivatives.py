from __future__ import annotations

import numpy as np
import pytest

from app.core.diagnostics import check_derivatives
from app.core.scaling import ParameterScaling
from app.parsing import CompiledModel
from app.status import SolverError, Status


def test_restricted_parser_rejects_unsafe_expression():
    with pytest.raises(SolverError) as exc:
        CompiledModel("__import__('os').system('id')", ["a"], 2)
    assert exc.value.status == Status.INVALID_EXPRESSION
    assert "mathematical functions" in exc.value.message


def test_forward_ad_matches_central_differences_for_nonlinear_model():
    x = np.linspace(0.1, 3.0, 12)
    model = CompiledModel("a*exp(-b*x)+sqrt(c+x)", ["a", "b", "c"], x.size)
    theta = np.array([2.0, 0.4, 1.0])
    scaling = ParameterScaling.create(
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([10.0, 5.0, 10.0]),
        scales=np.array([2.0, 0.1, 0.5]),
    )
    check = check_derivatives(
        model,
        scaling,
        scaling.to_internal(theta),
        x,
        step=np.sqrt(np.finfo(float).eps),
        rtol=1.0e-6,
        atol=1.0e-6,
        when="unit-test",
    )
    assert check.passed is True
    assert check.max_absolute_error < 1.0e-6
