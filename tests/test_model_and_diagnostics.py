from __future__ import annotations

import numpy as np
import pytest

from milp.diagnostics import verify_candidate
from milp.errors import InputError
from milp.model import parse_problem
from milp.schemas import LPProblem


def test_parser_flips_maximization_and_constraints():
    parsed = parse_problem(
        LPProblem.model_validate(
            {
                "name": "parse",
                "sense": "maximize",
                "objective": {"x": 2, "y": -1},
                "variables": [
                    {"name": "x", "type": "integer", "lower": -1.2, "upper": 3.9},
                    {"name": "y", "type": "continuous", "lower": 0, "upper": 5},
                ],
                "constraints": [{"name": "ge", "terms": {"x": 2, "y": -1}, "sense": ">=", "rhs": 1}],
            }
        )
    )

    assert parsed.objective.tolist() == [-2.0, 1.0]
    assert parsed.internal_lower[0] == -1.0
    assert parsed.internal_upper[0] == 3.0
    assert parsed.a_ub.toarray().tolist() == [[-2.0, 1.0]]
    assert parsed.b_ub.tolist() == [-1.0]


def test_integer_variables_must_have_finite_bounds():
    with pytest.raises(InputError) as excinfo:
        parse_problem(
            LPProblem.model_validate(
                {
                    "name": "unbounded-int",
                    "sense": "minimize",
                    "objective": {"x": 1},
                    "variables": [{"name": "x", "type": "integer", "lower": 0}],
                }
            )
        )
    assert "finite lower and upper bounds" in str(excinfo.value)


def test_candidate_violation_diagnostic_is_specific():
    parsed = parse_problem(
        LPProblem.model_validate(
            {
                "name": "diag",
                "sense": "minimize",
                "objective": {"x": 1},
                "variables": [{"name": "x", "type": "integer", "lower": 0, "upper": 2}],
                "constraints": [{"name": "eq", "terms": {"x": 1}, "sense": "==", "rhs": 1.0}],
            }
        )
    )
    check = verify_candidate(parsed, np.array([0.0]), integer_tolerance=1e-8, constraint_tolerance=1e-8)

    assert not check.feasible
    assert check.failure_category == "equality_violation"
    assert check.max_equality_residual == 1.0


def test_non_integer_bounds_leave_empty_integer_domain():
    parsed = parse_problem(
        LPProblem.model_validate(
            {
                "name": "empty-int-domain",
                "sense": "minimize",
                "objective": {"x": 1},
                "variables": [{"name": "x", "type": "integer", "lower": 0.2, "upper": 0.8}],
            }
        )
    )

    assert parsed.internal_lower[0] == 1.0
    assert parsed.internal_upper[0] == 0.0
