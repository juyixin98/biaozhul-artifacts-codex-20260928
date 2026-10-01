from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from app.core import solve_least_squares
from app.status import SolverError, Status


def test_exponential_bad_initial_matches_independent_scipy_reference(exponential_fixture, solver_config, run_logger):
    fixture = exponential_fixture
    result = solve_least_squares(
        model=fixture["model"],
        x=fixture["x"],
        y=fixture["y"],
        initial=fixture["initial"],
        lower=fixture["lower"],
        upper=fixture["upper"],
        scales=fixture["scales"],
        config=solver_config,
        logger=run_logger,
    )

    reference = least_squares(
        lambda p: fixture["y"] - (p[0] * np.exp(-p[1] * fixture["x"]) + p[2]),
        fixture["initial"],
        bounds=(fixture["lower"], fixture["upper"]),
    )

    assert result.status == Status.CONVERGED
    assert result.stationarity_type == "interior_stationary"
    np.testing.assert_allclose(result.parameters, reference.x, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        result.residuals,
        fixture["y"] - (reference.x[0] * np.exp(-reference.x[1] * fixture["x"]) + reference.x[2]),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert result.residual_sum_squares < 0.01
    assert result.derivative_checks[0]["when"] == "initial"
    assert result.derivative_checks[-1]["when"] == "final"
    assert result.identifiability["identifiable"] is True


def test_every_iteration_records_actual_and_predicted_reduction(exponential_fixture, solver_config, run_logger):
    result = solve_least_squares(
        model=exponential_fixture["model"],
        x=exponential_fixture["x"],
        y=exponential_fixture["y"],
        initial=exponential_fixture["initial"],
        lower=exponential_fixture["lower"],
        upper=exponential_fixture["upper"],
        scales=exponential_fixture["scales"],
        config=solver_config,
        logger=run_logger,
    )

    accepted = [row for row in result.iteration_history if row["accepted"]]
    assert len(accepted) >= 5
    assert any(row["reduction_ratio"] < 0.0 for row in result.iteration_history)
    for row in result.iteration_history:
        assert row["predicted_reduction"] > 0.0
        assert "actual_reduction" in row
        assert "decision" in row
    for row in accepted:
        assert row["actual_reduction"] > 0.0
        assert row["reduction_ratio"] >= solver_config.accept_ratio


def test_saturation_strong_correlation_still_identifiable_and_matches_reference(solver_config, run_logger):
    rng = np.random.default_rng(7)
    x = np.linspace(0.01, 0.1, 50)
    true = np.array([5.0, 2.0])
    y = true[0] * x / (true[1] + x) + rng.normal(0.0, 0.0001, x.size)

    from app.parsing import CompiledModel

    model = CompiledModel("v*x/(k+x)", ["v", "k"], x.size)
    result = solve_least_squares(
        model=model,
        x=x,
        y=y,
        initial=np.array([1.0, 8.0]),
        lower=np.array([0.0, 0.01]),
        upper=np.array([20.0, 20.0]),
        scales=np.array([1.0, 0.5]),
        config=solver_config,
        logger=run_logger,
    )
    reference = least_squares(
        lambda p: y - p[0] * x / (p[1] + x),
        np.array([1.0, 8.0]),
        bounds=([0.0, 0.01], [20.0, 20.0]),
    )

    assert result.status == Status.CONVERGED
    assert result.stationarity_type == "interior_stationary"
    assert abs(result.identifiability["parameter_correlation"][0][1]) > 0.9999
    assert result.identifiability["condition_number"] > 100.0
    np.testing.assert_allclose(result.parameters, reference.x, rtol=0.0, atol=1.0e-4)


def test_boundary_stationary_is_distinct_from_interior_stationary(solver_config, run_logger):
    rng = np.random.default_rng(42)
    x = np.linspace(0.1, 10.0, 40)
    y = 5.0 * x / (2.0 + x) + rng.normal(0.0, 0.005, x.size)
    from app.parsing import CompiledModel

    model = CompiledModel("v*x/(k+x)", ["v", "k"], x.size)
    result = solve_least_squares(
        model=model,
        x=x,
        y=y,
        initial=np.array([0.1, 15.0]),
        lower=np.array([0.0, 0.01]),
        upper=np.array([4.0, 20.0]),
        scales=np.array([2.0, 2.0]),
        config=solver_config.model_copy(update={"gradient_tol": 1.0e-6}),
        logger=run_logger,
    )

    assert result.status == Status.CONVERGED
    assert result.stationarity_type == "boundary_stationary"
    assert result.active_bounds == {"v": "upper"}
    assert result.parameters[0] == pytest.approx(4.0, abs=1.0e-8)


def test_nonconvergent_has_explicit_category_without_success_status(run_logger):
    from app.parsing import CompiledModel

    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 4.0, 30)
    y = 3.0 * np.exp(-0.7 * x) + 0.5 + rng.normal(0.0, 0.01, x.size)
    model = CompiledModel("a*exp(-b*x)+c", ["a", "b", "c"], x.size)
    result = solve_least_squares(
        model=model,
        x=x,
        y=y,
        initial=np.array([0.1, 2.5, 2.0]),
        lower=np.array([0.0, 0.0, -2.0]),
        upper=np.array([10.0, 5.0, 5.0]),
        scales=np.array([1.0, 0.2, 1.0]),
        config=solver_config_with(max_iterations=2),
        logger=run_logger,
    )

    assert result.status == Status.NONCONVERGENT
    assert "maximum iterations" in result.message
    assert result.stationarity_type is None
    assert result.diagnostics["stationary_according_to_tolerance"] is False


def test_rank_deficient_model_is_unidentifiable(run_logger):
    from app.parsing import CompiledModel

    x = np.linspace(0.0, 1.0, 12)
    y = 2.0 + 0.5 * x
    model = CompiledModel("a+b+c*x", ["a", "b", "c"], x.size)
    result = solve_least_squares(
        model=model,
        x=x,
        y=y,
        initial=np.zeros(3),
        lower=np.full(3, -10.0),
        upper=np.full(3, 10.0),
        scales=np.ones(3),
        config=solver_config_with(),
        logger=run_logger,
    )

    assert result.status == Status.UNIDENTIFIABLE
    assert result.identifiability["numerical_rank"] == 2
    assert result.identifiability["parameter_count"] == 3
    assert result.identifiability["parameter_correlation"][0][1] == pytest.approx(1.0, abs=1.0e-8)
    assert result.stationarity_type == "interior_stationary"


def test_illegal_initial_function_value_is_separate_error(run_logger):
    from app.parsing import CompiledModel

    x = np.linspace(0.1, 1.0, 10)
    model = CompiledModel("log(x-b)", ["b"], x.size)
    with pytest.raises(SolverError) as exc:
        solve_least_squares(
            model=model,
            x=x,
            y=np.ones_like(x),
            initial=np.array([2.0]),
            lower=np.array([-10.0]),
            upper=np.array([10.0]),
            scales=np.ones(1),
            config=solver_config_with(),
            logger=run_logger,
        )
    assert exc.value.status == Status.ILLEGAL_FUNCTION_VALUE
    assert exc.value.details["nonfinite_count"] == x.size
    assert "initial prediction" in exc.value.details["where"]


def test_illegal_candidate_is_recorded_and_trust_region_shrinks(monkeypatch, run_logger):
    from app.parsing import CompiledModel

    x = np.linspace(0.0, 1.0, 8)
    y = 2.0 * x + 3.0
    model = CompiledModel("a*x+b", ["a", "b"], x.size)
    original_evaluate = model.evaluate
    calls = {"count": 0}

    def guarded_evaluate(parameters, x_values):
        calls["count"] += 1
        if calls["count"] > 1:
            return np.full_like(x_values, np.nan)
        return original_evaluate(parameters, x_values)

    def guarded_jacobian(parameters, x_values):
        return np.column_stack((x_values, np.ones_like(x_values)))

    monkeypatch.setattr(model, "evaluate", guarded_evaluate)
    monkeypatch.setattr(model, "jacobian", guarded_jacobian)
    with pytest.raises(SolverError) as exc:
        solve_least_squares(
            model=model,
            x=x,
            y=y,
            initial=np.zeros(2),
            lower=np.full(2, -10.0),
            upper=np.full(2, 10.0),
            scales=np.ones(2),
            config=solver_config_with(max_iterations=10, derivative_check=False),
            logger=run_logger,
        )

    assert exc.value.status == Status.ILLEGAL_FUNCTION_VALUE
    assert exc.value.details["where"].startswith("iteration ")


def solver_config_with(**updates):
    from app.config import SolverConfig

    data = {"max_iterations": 100, "gradient_tol": 1.0e-7}
    data.update(updates)
    return SolverConfig(**data)
