from __future__ import annotations

import math

import pytest

from change_detection.errors import InvalidParameterError, InvalidSeriesError
from change_detection.statistics import StablePrefixStats


def test_stable_prefix_stats_return_exact_constant_and_block_sums() -> None:
    stats = StablePrefixStats([0.0, 0.0, 10.0, 10.0])

    assert stats.segment_mean(0, 2) == 0.0
    assert stats.segment_sse(0, 2) == 0.0
    assert stats.segment_mean(2, 4) == 10.0
    assert stats.segment_sse(2, 4) == 0.0

    assert stats.segment_mean(0, 4) == 5.0
    assert stats.segment_sse(0, 4) == pytest.approx(100.0)


def test_large_baseline_small_change_is_numerically_recoverable() -> None:
    stats = StablePrefixStats([1e9, 1e9, 1e9 + 3.0, 1e9 + 3.0])

    assert stats.segment_sse(0, 2) == 0.0
    assert stats.segment_sse(2, 4) == 0.0
    assert stats.segment_sse(0, 4) == pytest.approx(9.0)


def test_statistics_reject_non_finite_and_illegal_bounds() -> None:
    with pytest.raises(InvalidSeriesError) as exc_info:
        StablePrefixStats([1.0, math.nan])
    assert exc_info.value.category == "INVALID_SERIES"

    stats = StablePrefixStats([1.0, 2.0])
    with pytest.raises(InvalidParameterError) as exc_info:
        stats.segment_sse(1, 1)
    assert exc_info.value.category == "INVALID_PARAMETER"

