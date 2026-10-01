from __future__ import annotations

import numpy as np
import pytest

from app.diagnostics import RunLogger
from app.solver import SolveOptions, SolveStatus, solve_nonlinear_least_squares


EXPRESSION_EXP = "amp*exp(-rate*x)+floor"
EXPRESSION_SAT = "a*(1-exp(-k*x))/(1+b*(1-exp(-k*x)))"
EXPRESSION_ILLEGAL = "amp*sqrt(x-offset)"


def test_exponential_decay_bad_start_recovers_reference(x_data, noisy, log_file, read_jsonl):
    truth = {"amp": 2.0, "rate": 1.5, "floor": 0.3}
    y = truth["amp"] * np.exp(-truth["rate"] * x_data) + truth["floor"] + noisy
    logger = RunLogger(log_file, run_id="exp-bad-start")

    result = solve_nonlinear_least_squares(
        EXPRESSION_EXP,
        ["amp", "rate", "floor"],
        x_data,
        y,
        initial={"amp": 0.05, "rate": 0.01, "floor": 1.7},
        lower={"amp": 0.0, "rate": 0.0, "floor": 0.0},
        upper={"amp": 8.0, "rate": 8.0, "floor": 8.0},
        scales={"amp": 2.0, "rate": 1.5, "floor": 1.0},
        options=SolveOptions(max_iterations=80),
        logger=logger,
    )

    assert result.success is True
    assert result.status == SolveStatus.CONVERGED_INTERIOR
    assert result.stationary_type == "interior"
    assert result.boundary_parameters == []
    assert result.parameters["amp"] == pytest.approx(truth["amp"], abs=3e-3)
    assert result.parameters["rate"] == pytest.approx(truth["rate"], abs=4e-3)
    assert result.parameters["floor"] == pytest.approx(truth["floor"], abs=3e-3)
    assert np.sqrt(np.mean(np.asarray(result.residuals) ** 2)) < 8e-4
    assert result.diagnostics.finite_difference_max_error < 1e-7

    accepted = [item for item in result.diagnostics.iterations if item.accepted]
    assert len(accepted) >= 5
    for item in result.diagnostics.iterations:
        assert item.predicted_reduction > 0.0
        if item.accepted:
            assert item.actual_reduction > 0.0
            assert item.reduction_ratio == pytest.approx(
                item.actual_reduction / item.predicted_reduction, rel=1e-10
            )

    events = read_jsonl(log_file)
    assert {event["run_id"] for event in events} == {"exp-bad-start"}
    assert events[0]["event"] == "solve_started"
    assert events[-1]["event"] == "solve_finished"
    assert events[0]["versions"]["numpy"] == np.__version__
    assert any(event["event"] == "derivative_check" and event["passed"] for event in events)


def test_saturation_model_recovers_reference(x_data, noisy, log_file):
    x_saturation = np.linspace(0.0, 4.0, 24)
    truth = {"a": 1.2, "k": 0.8, "b": 0.2}
    approach = 1.0 - np.exp(-truth["k"] * x_saturation)
    y = truth["a"] * approach / (1.0 + truth["b"] * approach)
    # Allowing a small negative region for b avoids an artificial boundary KKT
    # point at b=0 on the way from the deliberately bad initial value.

    result = solve_nonlinear_least_squares(
        EXPRESSION_SAT,
        ["a", "k", "b"],
        x_saturation,
        y,
        initial={"a": 0.1, "k": 0.03, "b": 1.3},
        lower={"a": 0.0, "k": 0.0, "b": -2.0},
        upper={"a": 5.0, "k": 5.0, "b": 5.0},
        scales={"a": 1.0, "k": 0.8, "b": 0.5},
        options=SolveOptions(max_iterations=100, tolerance=1e-4),
        logger=RunLogger(log_file, run_id="saturation"),
    )

    assert result.success is True
    assert result.status == SolveStatus.CONVERGED_INTERIOR
    assert result.parameters["a"] == pytest.approx(truth["a"], abs=8e-3)
    assert result.parameters["k"] == pytest.approx(truth["k"], abs=1.2e-2)
    assert result.parameters["b"] == pytest.approx(truth["b"], abs=1.2e-2)
    assert np.max(np.abs(result.residuals)) < 1e-5


def test_strongly_correlated_parameters_are_fit_with_ill_conditioning_diagnostic(x_data):
    amp, decay, floor = 1.8, 0.7, 0.4
    short_x = np.linspace(0.0, 0.5, 20)
    y = amp * np.exp(-decay * short_x) + floor

    result = solve_nonlinear_least_squares(
        "amp*exp(-decay*x)+floor",
        ["amp", "decay", "floor"],
        short_x,
        y,
        initial={"amp": 1.0, "decay": 0.2, "floor": 0.7},
        lower={"amp": 0.0, "decay": 0.0, "floor": -2.0},
        upper={"amp": 5.0, "decay": 3.0, "floor": 3.0},
        options=SolveOptions(max_iterations=100, tolerance=1e-3),
        logger=RunLogger(run_id="correlated"),
    )

    assert result.success is True
    assert result.diagnostics.condition_number > 1000.0
    assert result.diagnostics.rank == 3
    assert result.parameters["amp"] == pytest.approx(amp, abs=2e-3)
    assert result.parameters["decay"] == pytest.approx(decay, abs=2e-3)
    assert result.parameters["floor"] == pytest.approx(floor, abs=2e-3)
    assert result.cost < 1e-8


def test_boundary_stationary_point_is_distinct_from_interior(x_data):
    # The unconstrained minimum of fitting a constant is mean(y); forcing floor
    # above it makes lowering floor the only useful first-order direction.
    y = 1.0 + 0.05 * x_data
    result = solve_nonlinear_least_squares(
        "floor+slope*x",
        ["floor", "slope"],
        x_data,
        y,
        initial={"floor": 2.0, "slope": 0.5},
        lower={"floor": 1.8, "slope": -2.0},
        upper={"floor": 3.0, "slope": 2.0},
        options=SolveOptions(),
        logger=RunLogger(run_id="boundary"),
    )

    assert result.success is True
    assert result.status == SolveStatus.CONVERGED_BOUNDARY
    assert result.stationary_type == "boundary"
    assert result.boundary_parameters == ["floor"]
    assert result.parameters["floor"] == pytest.approx(1.8, abs=1e-8)
    assert abs(result.parameters["slope"]) > 0.0


def test_max_iterations_has_distinct_status(x_data):
    y = 2.0 * np.exp(-1.5 * x_data) + 0.3
    result = solve_nonlinear_least_squares(
        EXPRESSION_EXP,
        ["amp", "rate", "floor"],
        x_data,
        y,
        initial={"amp": 0.05, "rate": 0.01, "floor": 1.7},
        lower={"amp": 0.0, "rate": 0.0, "floor": 0.0},
        upper={"amp": 8.0, "rate": 8.0, "floor": 8.0},
        scales={"amp": 2.0, "rate": 1.5, "floor": 1.0},
        options=SolveOptions(max_iterations=1),
        logger=RunLogger(run_id="max-iter"),
    )

    assert result.success is False
    assert result.status == SolveStatus.MAX_ITERATIONS
    assert result.iterations == 1
    assert result.accepted_iterations in (0, 1)
    assert "Maximum 1 iterations" in result.reason


def test_unidentifiable_parameter_has_numeric_diagnosis(x_data):
    result = solve_nonlinear_least_squares(
        "a+b+0*x",
        ["a", "b"],
        x_data,
        np.zeros_like(x_data),
        initial={"a": 0.2, "b": 0.3},
        lower={"a": -1.0, "b": -1.0},
        upper={"a": 1.0, "b": 1.0},
        options=SolveOptions(derivative_check=False),
        logger=RunLogger(run_id="unidentifiable"),
    )

    assert result.success is False
    assert result.status == SolveStatus.UNIDENTIFIABLE
    assert result.diagnostics.rank == 1
    assert result.diagnostics.singular_values is not None
    assert len(result.diagnostics.nullspace) == 1
    assert result.iterations == 0
    assert "rank 1" in result.reason


def test_illegal_function_value_has_distinct_status(x_data):
    result = solve_nonlinear_least_squares(
        EXPRESSION_ILLEGAL,
        ["amp", "offset"],
        x_data,
        np.zeros_like(x_data),
        initial={"amp": 1.0, "offset": 1.5},
        lower={"amp": 0.0, "offset": 0.0},
        upper={"amp": 3.0, "offset": 3.0},
        options=SolveOptions(derivative_check=False),
        logger=RunLogger(run_id="illegal"),
    )

    assert result.success is False
    assert result.status == SolveStatus.ILLEGAL_FUNCTION_VALUE
    assert "Illegal value" in result.reason
