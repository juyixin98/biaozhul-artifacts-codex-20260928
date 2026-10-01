from __future__ import annotations

import numpy as np
import pytest

from app.core.diagnostics import FitError, FitStatus
from app.core.expression import (
    ExpressionModel,
    RestrictedExpression,
    check_jacobian,
)


def test_automatic_derivative_matches_finite_difference_review():
    theta = np.array([2.0, 0.3, 0.4])
    x = np.linspace(0.0, 1.0, 5)
    expression = RestrictedExpression("a*exp(-b*x)+c", ["a", "b", "c"])
    model = ExpressionModel(expression, x, np.zeros_like(x))
    residual, jacobian = model.residual_and_jacobian(theta)
    report = check_jacobian(
        model.residual,
        jacobian,
        theta,
        (np.zeros(3), np.array([10.0, 5.0, 10.0])),
        ["a", "b", "c"],
    )

    assert report.passed is True
    assert report.max_relative_error < 1e-7
    assert np.all(np.isfinite(residual))


def test_derivative_mismatch_is_detected_when_analytical_jacobian_is_corrupted():
    theta = np.array([2.0, 0.3])
    x = np.linspace(0.0, 1.0, 5)
    expression = RestrictedExpression("a*exp(-b*x)", ["a", "b"])
    model = ExpressionModel(expression, x, np.zeros_like(x))
    _residual, jacobian = model.residual_and_jacobian(theta)
    corrupted = jacobian.copy()
    corrupted[:, 0] += 0.2

    report = check_jacobian(
        model.residual,
        corrupted,
        theta,
        (np.zeros(2), np.array([10.0, 5.0])),
        ["a", "b"],
        tolerance=1e-8,
    )
    assert report.passed is False
    assert report.worst_parameter == "a"
    assert report.max_relative_error > 0.05


def test_restricted_expression_blocks_non_math_constructs():
    with pytest.raises(FitError) as exc_info:
        RestrictedExpression("__import__('os').system('x')", ["a"])
    assert exc_info.value.status == FitStatus.INVALID_EXPRESSION


def test_unknown_identifier_is_an_expression_error():
    with pytest.raises(FitError) as exc_info:
        RestrictedExpression("a*x+z", ["a"])
    assert exc_info.value.status == FitStatus.INVALID_EXPRESSION


def test_non_finite_initial_model_value_remains_illegal_value_category():
    expression = RestrictedExpression("a*log(x)", ["a"])
    model = ExpressionModel(expression, np.array([0.0, 1.0]), np.zeros(2))
    with pytest.raises(FitError) as exc_info:
        model.residual_and_jacobian(np.array([1.0]))
    assert exc_info.value.status == FitStatus.INVALID_INITIAL_RESIDUAL
