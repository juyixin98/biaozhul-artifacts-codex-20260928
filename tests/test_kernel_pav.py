from __future__ import annotations

import math

import pytest

from app.kernel import aggregate_equal_scores, fit_pav
from app.validation import CleanPoint


def p(score: float, target: int, weight: float = 1.0) -> CleanPoint:
    return CleanPoint(score=score, target=target, weight=weight, record_id=None)


def test_hand_computed_violation_pools_all_adjacent_blocks():
    # Scores 1..4 have means 1, 1, 0, 0. The first violation repeatedly makes
    # the accumulated block infeasible with its left neighbor.
    result = fit_pav([p(1, 1), p(2, 1), p(3, 0), p(4, 0)])

    assert result.scores == (1.0,)
    assert result.probabilities == (pytest.approx((0.5,)))
    assert [event.sequence for event in result.merge_events] == [1, 2, 3]
    assert result.merge_events[0].lower_score == 2.0
    assert result.merge_events[0].upper_score == 3.0
    assert result.merge_events[0].merged_probability == pytest.approx(0.5)
    assert result.merge_events[1].lower_score == 1.0
    assert result.merge_events[1].upper_score == 3.0
    assert result.merge_events[1].merged_probability == pytest.approx(2 / 3)
    assert result.merge_events[2].lower_score == 1.0
    assert result.merge_events[2].upper_score == 4.0
    assert result.merge_events[2].merged_probability == pytest.approx(0.5)


def test_repeated_scores_are_merged_by_weight_before_pav():
    grouped = aggregate_equal_scores(
        [p(2, 0, 2.0), p(2, 1, 6.0), p(1, 1, 3.0), p(1, 0, 1.0)]
    )

    assert [(item.score, item.target, item.weight) for item in grouped] == [
        (1.0, 0.75, 4.0),
        (2.0, 0.75, 8.0),
    ]

    result = fit_pav([p(2, 0, 2.0), p(2, 1, 6.0), p(1, 1, 3.0), p(1, 0, 1.0)])
    assert result.scores == (1.0, 2.0)
    assert result.probabilities == (pytest.approx((0.75, 0.75)))
    assert result.merge_events == ()


def test_boundary_weights_choose_expected_block_and_loss():
    points = [p(0.0, 0, 1000.0), p(1.0, 1, 1.0), p(2.0, 0, 1.0)]
    result = fit_pav(points)

    assert result.scores == (0.0, 1.0)
    assert result.probabilities == (pytest.approx((0.0, 0.5)))

    # Independent weighted SSE: sum w*y - (sum w*y)^2 / sum w for each block.
    expected_sse = 0.5
    assert result.weighted_sse == pytest.approx(expected_sse)

    assert result.merge_events[0].left_weight == 1.0
    assert result.merge_events[0].right_weight == 1.0
    assert result.merge_events[0].weighted_loss_delta == pytest.approx(0.5)
    assert len(result.merge_events) == 1


def test_zero_weight_is_excluded_and_does_not_create_score_boundary():
    points = [p(0.0, 1, 0.0), p(1.0, 1, 1.0), p(2.0, 0, 1.0)]

    # Validation is responsible for zero-weight filtering; feed the effective
    # positive-weight training set to the kernel.
    result = fit_pav([point for point in points if point.weight > 0])

    assert result.scores == (1.0,)
    assert result.probabilities == (pytest.approx((0.5,)))
    assert math.isfinite(result.weighted_sse)
