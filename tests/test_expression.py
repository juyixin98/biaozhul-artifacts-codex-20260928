from __future__ import annotations

import numpy as np
import pytest

from app.expression import ExpressionModel, ExpressionValidationError, IllegalFunctionError


def test_automatic_derivatives_match_independent_differences():
    x = np.linspace(0.0, 1.0, 5)
    model = ExpressionModel("a*exp(-b*x)/(1+c*x)", ["a", "b", "c"])
    params = {"a": 1.4, "b": 0.8, "c": 0.3}
    values, jacobian = model.evaluate(x, params)

    assert values.shape == (5,)
    assert jacobian.shape == (5, 3)
    h = 1e-7
    for j, name in enumerate(model.parameter_names):
        bumped = dict(params)
        bumped[name] += h
        numeric = (model.evaluate(x, bumped)[0] - values) / h
        assert jacobian[:, j] == pytest.approx(numeric, rel=1e-6, abs=1e-7)


def test_unsafe_expression_is_rejected_before_evaluation():
    with pytest.raises(ExpressionValidationError, match="Unsupported syntax"):
        ExpressionModel("__import__('os').system('echo bad')", ["unused"])


def test_unknown_name_is_rejected():
    with pytest.raises(ExpressionValidationError, match="Unknown name"):
        ExpressionModel("a + unknown", ["a"])


def test_illegal_domain_is_specific_error():
    model = ExpressionModel("sqrt(x-offset)", ["offset"])
    with pytest.raises(IllegalFunctionError, match="NaN or infinity"):
        model.evaluate(np.array([0.0, 1.0]), {"offset": 2.0})
