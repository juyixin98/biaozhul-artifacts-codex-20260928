"""Deterministic one-to-many temporal matching.

The sparse series supplies one anchor per output pair. Every dense sample is
assigned to at most one anchor by nearest target time. Several dense samples may
feed one anchor, but their mean creates exactly one pair. Neither a dense
observation nor an anchor is repeated across pairs.
"""

from __future__ import annotations

import numpy as np

from .models import PairDetail


def match_pairs(
    anchor_times: np.ndarray,
    dense_times: np.ndarray,
    dense_values: np.ndarray,
    lag: float,
    tolerance: float,
    anchor_side: str,
) -> tuple[tuple[PairDetail, ...], int]:
    """Return anchor/dense pairs and number of unmatched dense observations."""
    target_times = anchor_times - lag
    insertion = np.searchsorted(target_times, dense_times, side="left")

    nearest_anchor = np.full(dense_times.shape, -1, dtype=np.int64)
    nearest_distance = np.full(dense_times.shape, np.inf, dtype=np.float64)

    for dense_index, dense_time in enumerate(dense_times):
        right = int(insertion[dense_index])
        left = right - 1
        candidates: list[int] = []
        if left >= 0:
            candidates.append(left)
        if right < len(target_times):
            candidates.append(right)

        for anchor_index in candidates:
            distance = abs(float(dense_time) - float(target_times[anchor_index]))
            if distance < nearest_distance[dense_index] or (
                distance == nearest_distance[dense_index]
                and anchor_index < nearest_anchor[dense_index]
            ):
                nearest_distance[dense_index] = distance
                nearest_anchor[dense_index] = anchor_index

    grouped: dict[int, list[int]] = {}
    for dense_index, anchor_index in enumerate(nearest_anchor):
        if nearest_distance[dense_index] <= tolerance:
            grouped.setdefault(int(anchor_index), []).append(dense_index)

    pairs: list[PairDetail] = []
    for anchor_index in sorted(grouped):
        dense_indices = tuple(grouped[anchor_index])
        paired_time = float(np.mean(dense_times[list(dense_indices)]))
        anchor_time = float(anchor_times[anchor_index])
        pairs.append(
            PairDetail(
                anchor_side=anchor_side,  # type: ignore[arg-type]
                anchor_time=anchor_time,
                paired_time=paired_time,
                anchor_index=anchor_index,
                paired_indices=dense_indices,
                time_delta=anchor_time - paired_time,
                lag=lag,
            )
        )

    used = sum(len(indices) for indices in grouped.values())
    return tuple(pairs), int(len(dense_times) - used)
