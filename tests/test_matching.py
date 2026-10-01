from __future__ import annotations

import numpy as np

from lag_correlation.kernel.matching import match_pairs


def test_nearest_assignment_and_tie_do_not_repeat_dense_samples() -> None:
    pairs, unmatched = match_pairs(
        anchor_times=np.array([4.0, 6.0]),
        dense_times=np.array([5.0, 6.0, 9.0]),
        dense_values=np.array([10.0, 11.0, 12.0]),
        lag=0.0,
        tolerance=1.0,
        anchor_side="x",
    )

    assert unmatched == 1
    assert len(pairs) == 2
    assert pairs[0].anchor_index == 0
    assert pairs[0].paired_indices == (0,)
    assert pairs[1].anchor_index == 1
    assert pairs[1].paired_indices == (1,)

    used = [index for pair in pairs for index in pair.paired_indices]
    assert len(used) == len(set(used))


def test_multiple_dense_samples_feed_one_anchor_only_once() -> None:
    pairs, unmatched = match_pairs(
        anchor_times=np.array([10.0, 20.0]),
        dense_times=np.array([9.8, 10.0, 10.2]),
        dense_values=np.array([1.0, 2.0, 3.0]),
        lag=0.0,
        tolerance=0.25,
        anchor_side="y",
    )

    assert unmatched == 0
    assert len(pairs) == 1
    assert pairs[0].anchor_index == 0
    assert pairs[0].paired_indices == (0, 1, 2)
    assert pairs[0].paired_time == 10.0
    assert pairs[0].anchor_side == "y"
