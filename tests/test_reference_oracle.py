from __future__ import annotations

from fractions import Fraction

import pytest

from change_detection.errors import InvalidParameterError
from change_detection.reference import enumerate_best_segmentation


def test_enumeration_counts_all_legal_segmentations() -> None:
    result = enumerate_best_segmentation([0, 0, 10, 10], 1, 0)

    assert result.exact is True
    assert result.considered_segmentations == 8
    assert result.breakpoints == (2,)
    assert result.objective == Fraction(0)
    assert result.sse == Fraction(0)


def test_enumeration_enforces_minimum_segment_length() -> None:
    result = enumerate_best_segmentation([0, 0, 10, 10], 2, 0)

    assert result.considered_segmentations == 2
    assert result.breakpoints == (2,)
    assert [segment.cost for segment in result.segments] == [Fraction(0), Fraction(0)]


def test_float_oracle_uses_independent_scipy_norm() -> None:
    result = enumerate_best_segmentation([0.0, 0.0, 3.0, 3.0], 1, 1.0)

    assert result.exact is False
    assert result.considered_segmentations == 8
    assert result.breakpoints == (2,)
    assert result.objective == pytest.approx(2.0)


def test_reference_rejects_negative_penalty_with_named_category() -> None:
    with pytest.raises(InvalidParameterError) as exc_info:
        enumerate_best_segmentation([1, 2], 1, -1)
    assert exc_info.value.category == "INVALID_PARAMETER"

