from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_least_squares

from app.core.diagnostics import FitError, FitStatus
from app.core.expression import ExpressionModel, RestrictedExpression
from app.core.solver import solve_least_squares


def fit_expression(
    source: str,
    names: list[str],
    x,
    y,
    initial,
    lower,
    upper,
    *,
    scales=None,
    **kwargs,
):
    expression = RestrictedExpression(source, names)
    model = ExpressionModel(expression, x, y)
    return solve_least_squares(
        model.residual,
        model.residual_and_jacobian,
        np.asarray(initial, dtype=float),
        (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        scales=np.asarray(scales, dtype=float) if scales is not None else None,
        parameter_names=names,
        **kwargs,
    )


def test_exponential_decay_recovers_independent_reference_and_records_decreases(
    exponential_data,
):
    x, y, true = exponential_data
    initial = [0.05, 3.0, 0.001]
    lower = [0.0, 0.0, 0.0]
    upper = [10.0, 5.0, 10.0]

    result = fit_expression(
        "a*exp(-b*x)+c",
        ["a", "b", "c"],
        x,
        y,
        initial,
        lower,
        upper,
    )

    reference = scipy_least_squares(
        lambda theta: theta[0] * np.exp(-theta[1] * x) + theta[2] - y,
        np.asarray(initial),
        bounds=(lower, upper),
        method="trf",
    )

    assert result.success is True
    assert result.status in {
        FitStatus.OPTIMAL,
        FitStatus.CONVERGED_COST_TOLERANCE,
    }
    assert result.stationarity_type == "interior"
    assert result.parameter_map() == pytest.approx(true, abs=2e-4)
    assert result.x == pytest.approx(reference.x, abs=1e-6)
    assert np.max(np.abs(result.residuals)) == pytest.approx(
        np.max(np.abs(reference.fun)), abs=1e-5
    )
    assert result.cost == pytest.approx(0.5 * np.dot(reference.fun, reference.fun), abs=1e-8)
    assert result.projected_gradient_norm < 1e-7

    accepted = [item for item in result.iterations if item.accepted]
    assert accepted
    for item in accepted:
        assert item.actual_reduction > 0.0
        assert item.predicted_reduction > 0.0
        assert item.reduction_ratio == pytest.approx(
            item.actual_reduction / item.predicted_reduction, rel=1e-10
        )
        assert np.isfinite(item.radius)
        assert item.step_norm_scaled > 0.0


def test_saturation_model_with_bad_initial_and_correlated_parameters(saturation_data):
    x, y, true = saturation_data
    # At 0.01 the initial response is almost zero, giving a strongly curved start.
    initial = [0.01, 2.0]
    lower = [1.0e-6, 1.0e-6]
    upper = [10.0, 5.0]
    result = fit_expression(
        "A*x/(K+x)",
        ["A", "K"],
        x,
        y,
        initial,
        lower,
        upper,
        scales=[1.0, 0.1],
    )

    reference = scipy_least_squares(
        lambda theta: theta[0] * x / (theta[1] + x) - y,
        np.asarray(initial),
        bounds=(lower, upper),
        method="trf",
        x_scale=[1.0, 0.1],
    )

    assert result.success is True
    assert result.x == pytest.approx([true["A"], true["K"]], abs=2e-4)
    assert result.x == pytest.approx(reference.x, abs=1e-6)
    assert np.linalg.norm(result.residuals) == pytest.approx(
        np.linalg.norm(reference.fun), rel=1e-7
    )
    assert result.identifiability.identifiable is True
    assert abs(result.identifiability.parameter_correlation[0][1]) > 0.93


def test_rank_deficient_model_is_not_reported_as_success(collinear_data):
    x, y = collinear_data
    result = fit_expression(
        "a*x+b*x",
        ["a", "b"],
        x,
        y,
        [2.0, -1.0],
        [-10.0, -10.0],
        [10.0, 10.0],
    )

    assert result.success is False
    assert result.status == FitStatus.UNIDENTIFIABLE
    assert result.identifiability.identifiable is False
    assert result.identifiability.rank == 1
    assert result.identifiability.parameter_count == 2
    assert np.linalg.norm(result.residuals) < 1e-4


def test_boundary_stationary_point_is_distinct_from_interior_point():
    x = np.linspace(0.0, 1.0, 20)
    y = np.ones_like(x)
    result = fit_expression(
        "a+b*x",
        ["a", "b"],
        x,
        y,
        [0.0, 0.0],
        [-10.0, -10.0],
        [0.0, 10.0],
    )

    assert result.success is True
    assert result.status == FitStatus.OPTIMAL
    assert result.stationarity_type == "mixed"
    active_lower, active_upper = result.active_map()
    assert active_lower == {"a": False, "b": False}
    assert active_upper == {"a": True, "b": False}
    assert result.x == pytest.approx([0.0, 1.4615384615384617], abs=1e-10)


def test_iteration_limit_returns_non_convergence_not_success(exponential_data):
    x, y, _ = exponential_data
    result = fit_expression(
        "a*exp(-b*x)+c",
        ["a", "b", "c"],
        x,
        y,
        [0.05, 3.0, 0.001],
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 10.0],
        max_iterations=1,
    )

    assert result.success is False
    assert result.status == FitStatus.NON_CONVERGED_MAX_ITERATIONS
    assert "iteration limit" in result.message
    assert len(result.iterations) == 1


def test_illegal_initial_value_has_dedicated_diagnosis():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.0])
    with pytest.raises(FitError) as exc_info:
        fit_expression(
            "a*log(x)",
            ["a"],
            x,
            y,
            [1.0],
            [0.0],
            [10.0],
        )
    assert exc_info.value.status == FitStatus.INVALID_INITIAL_RESIDUAL


def test_illegal_trial_value_has_dedicated_diagnosis():
    x = np.array([5.0])
    y = np.array([-1.0])
    result = fit_expression(
        "sqrt(x-a)",
        ["a"],
        x,
        y,
        [4.9],
        [-100.0],
        [100.0],
        max_invalid_trials=0,
    )

    assert result.success is False
    assert result.status == FitStatus.INVALID_TRIAL_VALUE
    assert result.invalid_evaluations == 1
    assert result.iterations[0].judgment.startswith("rejected: illegal function value")


def test_explicit_scales_keep_large_and_small_parameter_steps_meaningful(exponential_data):
    x, y, true = exponential_data
    result = fit_expression(
        "a*exp(-b*x)+c",
        ["a", "b", "c"],
        x,
        y,
        [0.05, 3.0, 0.001],
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 10.0],
        scales=[5.0, 1.0, 0.5],
    )
    assert result.success is True
    assert result.scales == pytest.approx([5.0, 1.0, 0.5])
    assert result.x == pytest.approx([true["a"], true["b"], true["c"]], abs=2e-4)
