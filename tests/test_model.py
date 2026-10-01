"""Model parsing: structural validation and failure categories."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bilp.config import SolverConfig
from bilp.errors import InputValidationError, ResourceExhaustedError
from bilp.model import parse_problem
from bilp.schemas import ProblemSpec

CFG = SolverConfig()


def parse(problem: dict, config: SolverConfig = CFG):
    return parse_problem(ProblemSpec(**problem), config)


def test_minimal_problem_parses_with_defaults():
    p = parse({"c": [1.0, 2.0], "bounds": [[0, 3], [0, 3]]})
    assert p.n == 2
    assert p.integer_idx.tolist() == [0, 1]  # default: all integer
    assert not p.maximize
    assert p.A_ub.shape == (0, 2)


def test_maximize_sense_parsed():
    p = parse({"sense": "maximize", "c": [1.0], "bounds": [[0, 1]]})
    assert p.maximize


def test_empty_objective_rejected():
    with pytest.raises(InputValidationError, match="non-empty"):
        parse({"c": []})


def test_nonfinite_objective_rejected():
    with pytest.raises(InputValidationError, match="non-finite"):
        parse({"c": [1.0, float("nan")], "bounds": [[0, 1], [0, 1]]})


def test_integrality_length_mismatch():
    with pytest.raises(InputValidationError, match="integrality"):
        parse({"c": [1.0, 2.0], "integrality": [1], "bounds": [[0, 1], [0, 1]]})


def test_integrality_value_out_of_range():
    with pytest.raises(InputValidationError, match="0 \\(continuous\\) or 1"):
        parse({"c": [1.0], "integrality": [2], "bounds": [[0, 1]]})


def test_bounds_length_mismatch():
    with pytest.raises(InputValidationError, match="bounds"):
        parse({"c": [1.0, 2.0], "bounds": [[0, 1]]})


def test_lower_bound_above_upper_rejected():
    with pytest.raises(InputValidationError, match="exceeds upper bound"):
        parse({"c": [1.0], "bounds": [[2, 1]]})


def test_integer_variable_must_be_bounded():
    with pytest.raises(InputValidationError, match="finite lower and upper"):
        parse({"c": [1.0], "bounds": [[0, None]]})


def test_continuous_variable_may_be_unbounded():
    p = parse({"c": [1.0], "integrality": [0], "bounds": [[None, None]]})
    assert p.integer_idx.size == 0


def test_a_ub_row_length_mismatch():
    with pytest.raises(InputValidationError, match=r"A_ub\[0\]"):
        parse({"c": [1.0, 2.0], "bounds": [[0, 1], [0, 1]],
               "A_ub": [[1.0]], "b_ub": [1.0]})


def test_a_ub_and_b_ub_must_come_in_pairs():
    with pytest.raises(InputValidationError, match="together"):
        parse({"c": [1.0], "bounds": [[0, 1]], "A_ub": [[1.0]]})


def test_b_ub_length_mismatch():
    with pytest.raises(InputValidationError, match="rows"):
        parse({"c": [1.0], "bounds": [[0, 1]],
               "A_ub": [[1.0], [1.0]], "b_ub": [1.0]})


def test_equality_constraints_parse():
    p = parse({"c": [1.0], "bounds": [[0, 2]],
               "A_eq": [[2.0]], "b_eq": [3.0]})
    assert p.A_eq.shape == (1, 1)
    assert p.b_eq.tolist() == [3.0]


def test_too_many_variables_is_resource_error():
    cfg = SolverConfig(max_variables=1)
    with pytest.raises(ResourceExhaustedError):
        parse({"c": [1.0, 2.0], "bounds": [[0, 1], [0, 1]]}, config=cfg)


def test_too_many_constraints_is_resource_error():
    cfg = SolverConfig(max_constraints=1)
    with pytest.raises(ResourceExhaustedError):
        parse({"c": [1.0], "bounds": [[0, 1]],
               "A_ub": [[1.0], [2.0]], "b_ub": [1.0, 2.0]}, config=cfg)


def test_unknown_field_rejected_by_schema():
    with pytest.raises(ValidationError):
        ProblemSpec(**{"c": [1.0], "bounds": [[0, 1]], "bogus": 1})
