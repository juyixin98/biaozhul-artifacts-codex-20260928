import numpy as np
import pytest

from entropic_ot.errors import InputValidationError
from entropic_ot.validation import validate_problem


def test_accepts_non_negative_masses_and_marks_zero_rows_and_columns():
    problem = validate_problem(
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [[0, 1, 2], [3, 4, 5], [6, 7, 0]],
        epsilon=0.1,
        max_iterations=100,
    )

    assert problem.active_rows.tolist() == [False, True, True]
    assert problem.active_columns.tolist() == [True, False, True]
    assert problem.active_cost.shape == (2, 2)


@pytest.mark.parametrize(
    ("source", "target", "field"),
    [
        ([-0.1, 1.1], [0.5, 0.5], "source_mass"),
        ([0.5, 0.5], [0.4, 0.5], "mass_balance"),
        ([np.nan, 1.0], [0.5, 0.5], "source_mass"),
        ([0.5, 0.5], [0.5, np.inf], "target_mass"),
    ],
)
def test_rejects_invalid_or_mismatched_masses(source, target, field):
    with pytest.raises(InputValidationError) as exc_info:
        validate_problem(
            source,
            target,
            [[0.0, 1.0], [1.0, 0.0]],
            epsilon=0.1,
            max_iterations=10,
        )

    assert exc_info.value.category == "input_error"
    assert exc_info.value.details["field"] == field


def test_rejects_shape_mismatch_and_non_finite_cost():
    with pytest.raises(InputValidationError) as shape_error:
        validate_problem([1, 1], [1, 1, 1], [[0, 1], [1, 0]], epsilon=0.1, max_iterations=10)
    assert shape_error.value.details["cost_shape"] == [2, 2]

    with pytest.raises(InputValidationError) as cost_error:
        validate_problem(
            [1, 1],
            [1, 1],
            [[0, np.inf], [1, 0]],
            epsilon=0.1,
            max_iterations=10,
        )
    assert cost_error.value.details["field"] == "cost"


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_epsilon(epsilon):
    with pytest.raises(InputValidationError) as exc_info:
        validate_problem([1], [1], [[0]], epsilon=epsilon, max_iterations=10)
    assert exc_info.value.category == "input_error"
    assert exc_info.value.details["field"] == "epsilon"


def test_rejects_non_positive_integer_iteration_budget():
    with pytest.raises(InputValidationError) as exc_info:
        validate_problem([1], [1], [[0]], epsilon=0.1, max_iterations=0)
    assert exc_info.value.category == "input_error"
    assert exc_info.value.details["field"] == "max_iterations"
