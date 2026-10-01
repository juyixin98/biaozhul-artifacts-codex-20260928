from __future__ import annotations

import pytest

from app.evidence import build_evidence, numerical_diagnostics
from app.kernel import CandidateInput, IncrementalSorter, Objective
from app.parsing import parse_candidates, parse_objectives, parse_problem, parse_tolerance


OBJECTIVES = (Objective("cost", "min"), Objective("risk", "min"))


def test_missing_objective_has_dedicated_failure_category():
    payload = {
        "objectives": [
            {"name": "cost", "direction": "min"},
            {"name": "risk", "direction": "min"},
        ],
        "candidates": [{"id": "A", "values": {"cost": 1}}],
    }
    with pytest.raises(Exception) as exc_info:
        parse_problem(payload)
    assert exc_info.value.code == "MISSING_OBJECTIVE"
    assert exc_info.value.details["missing_objectives"] == ["risk"]


def test_boolean_and_non_finite_values_are_rejected_before_ranking():
    with pytest.raises(Exception) as exc_info:
        parse_candidates(
            [{"id": "A", "values": {"cost": True, "risk": 1}}],
            OBJECTIVES,
        )
    assert exc_info.value.code == "NON_NUMERIC_OBJECTIVE"

    with pytest.raises(Exception) as exc_info:
        parse_candidates(
            [{"id": "A", "values": {"cost": 1, "risk": float("nan")}}],
            OBJECTIVES,
        )
    assert exc_info.value.code == "NON_FINITE_OBJECTIVE"


def test_unknown_and_duplicate_names_are_rejected():
    with pytest.raises(Exception) as exc_info:
        parse_objectives(
            [
                {"name": "cost", "direction": "min"},
                {"name": "cost", "direction": "max"},
            ]
        )
    assert exc_info.value.code == "DUPLICATE_OBJECTIVE_NAME"

    with pytest.raises(Exception) as exc_info:
        parse_candidates(
            [{"id": "A", "values": {"cost": 1, "risk": 2, "extra": 3}}],
            OBJECTIVES,
        )
    assert exc_info.value.code == "UNKNOWN_OBJECTIVE"

    with pytest.raises(Exception) as exc_info:
        parse_candidates(
            [
                {"id": "A", "values": {"cost": 1, "risk": 2}},
                {"id": "A", "values": {"cost": 2, "risk": 1}},
            ],
            OBJECTIVES,
        )
    assert exc_info.value.code == "DUPLICATE_CANDIDATE_ID"


def test_tolerance_does_not_enter_ordering_but_is_reported_as_uncertainty():
    with pytest.raises(Exception) as exc_info:
        parse_tolerance(-0.1)
    assert exc_info.value.code == "INVALID_TOLERANCE"

    sorter = IncrementalSorter(OBJECTIVES)
    sorter.add_candidates(
        [
            CandidateInput("A", (0.0, 0.0)),
            CandidateInput("B", (0.05, 0.05)),
        ]
    )
    evidence = build_evidence(sorter)
    diagnostics = numerical_diagnostics(sorter, 0.1)

    assert evidence.ordering_predicate.startswith("exact ")
    assert evidence.tolerance_applied_to_ordering is False
    assert evidence.independent_reference_layers == (("A",), ("B",))
    assert diagnostics.near_equal_pairs[0]["left_id"] == "A"
    assert diagnostics.uncertainties[0]["code"] == "NEAR_EQUAL_WITHIN_TOLERANCE"


def test_evidence_reports_duplicates_constant_dimension_and_graph_integrity():
    sorter = IncrementalSorter(
        (
            Objective("cost", "min"),
            Objective("constant", "min"),
        )
    )
    sorter.add_candidates(
        [
            CandidateInput("A", (1.0, 3.0)),
            CandidateInput("A2", (1.0, 3.0)),
            CandidateInput("B", (2.0, 3.0)),
        ]
    )

    evidence = build_evidence(sorter)
    diagnostics = numerical_diagnostics(sorter, 0.0)
    assert evidence.comparisons_performed == 3
    assert evidence.pairwise_dominance_edges == 2
    assert evidence.duplicate_groups == (("A", "A2"),)
    assert evidence.independent_reference_matched is True
    assert evidence.layer_non_dominated is True
    assert evidence.transitivity_preserved is True
    assert diagnostics.constant_objectives == ("constant",)
