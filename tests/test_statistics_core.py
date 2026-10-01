import numpy as np
import pytest

from changepoint_api.statistics_core import (
    build_prefix_stats,
    segment_sse,
    transition_sse,
    verified_segment,
)


def test_prefix_statistics_remain_stable_for_large_offset_and_change():
    samples = [1.0e9, 1.0e9, 1.0e9 + 4.0, 1.0e9 + 4.0]
    stats = build_prefix_stats(samples)

    assert float(segment_sse(stats, 0, 2)[0]) == pytest.approx(0.0, abs=1e-18)
    assert float(segment_sse(stats, 2, 4)[0]) == pytest.approx(0.0, abs=1e-18)
    assert float(segment_sse(stats, 0, 4)[0]) == pytest.approx(16.0, rel=1e-10)
    assert float(
        transition_sse(stats, np.array([0, 1, 2, 3]), 4)[0]
    ) == pytest.approx(16.0, rel=1e-10)
    )


def test_verified_segment_cross_checks_three_sse_views():
    samples = [1.0, 2.0, 3.0, 4.0]
    stats = build_prefix_stats(samples)

    mean, cost, prefix_cost, scipy_cost = verified_segment(stats, 0, 4)

    assert mean == pytest.approx(2.5)
    assert cost == pytest.approx(5.0)
    assert prefix_cost == pytest.approx(5.0)
    assert scipy_cost == pytest.approx(5.0)
