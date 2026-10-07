from __future__ import annotations

import math

import pytest

from app.errors import ValidationFailure
from app.validation import validate_records


def codes(exc: ValidationFailure) -> list[str]:
    assert isinstance(exc.details, list)
    return [item["code"] for item in exc.details]


def test_validation_uses_specific_failure_categories():
    with pytest.raises(ValidationFailure) as exc_info:
        validate_records(
            [
                {"score": None, "target": 1},
                {"score": float("nan"), "target": 0},
                {"score": 1, "target": 2},
                {"score": 1, "target": True},
                {"score": 1, "target": 0, "weight": -0.1},
                {"score": 1, "target": 0, "weight": float("inf")},
                "not-an-object",
            ]
        )

    assert codes(exc_info.value) == [
        "invalid_score",
        "non_finite_score",
        "non_binary_target",
        "invalid_target",
        "negative_weight",
        "non_finite_weight",
        "invalid_record",
    ]


def test_missing_and_explicit_null_weight_default_to_one():
    points, warnings = validate_records(
        [
            {"score": 0.0, "target": 0},
            {"score": 1.0, "target": 1, "weight": None},
        ]
    )

    assert warnings == []
    assert [point.weight for point in points] == [1.0, 1.0]


def test_zero_weight_is_warned_and_filtered():
    points, warnings = validate_records(
        [
            {"score": 0.0, "target": 1, "weight": 0.0, "record_id": "ignored"},
            {"score": 1.0, "target": 0, "weight": 1.0},
        ]
    )

    assert len(points) == 1
    assert points[0].score == 1.0
    assert warnings[0].code == "zero_weight"
    assert warnings[0].record_id == "ignored"


def test_empty_training_set_is_explicit_failure():
    with pytest.raises(ValidationFailure) as exc_info:
        validate_records([])
    assert codes(exc_info.value) == ["empty_records"]


def test_only_zero_weight_training_records_fails():
    with pytest.raises(ValidationFailure) as exc_info:
        validate_records([{"score": 1.0, "target": 0, "weight": 0.0}])
    assert codes(exc_info.value) == ["no_effective_records"]


def test_non_finite_float_is_rejected(math_sentinel: float = math.inf):
    with pytest.raises(ValidationFailure) as exc_info:
        validate_records([{"score": math_sentinel, "target": 0}])
    assert codes(exc_info.value) == ["non_finite_score"]
