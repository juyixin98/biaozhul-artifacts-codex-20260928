"""Weighted Pool-Adjacent-Violators (PAV) isotonic regression core.

Solves:  min_g  sum_i w_i * (y_i - g_i)^2   s.t. g_1 <= g_2 <= ... <= g_n
for observations already sorted by score. The result is a step function
described by blocks (contiguous runs sharing one fitted value).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PavBlock:
    """One level of the fitted step function."""

    start: int  # first index (inclusive) in the sorted input
    end: int    # last index (exclusive)
    weight: float
    value: float


@dataclass
class PavResult:
    blocks: list[PavBlock] = field(default_factory=list)

    @property
    def values(self) -> np.ndarray:
        """Fitted value per input observation (length n)."""
        out = np.empty(sum(b.end - b.start for b in self.blocks))
        for b in self.blocks:
            out[b.start:b.end] = b.value
        return out

    @property
    def weighted_loss(self) -> float:
        return sum(b.weight * b.value * b.value for b in self.blocks)


def pav(y: np.ndarray, w: np.ndarray) -> PavResult:
    """Weighted PAV. y, w must be 1-D, equal length, w > 0, no NaN.

    Returns blocks in input order with non-decreasing values.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.ndim != 1 or w.ndim != 1 or y.shape != w.shape:
        raise ValueError("y and w must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValueError("empty input")
    if not (np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError("y and w must be finite")
    if (w <= 0).any():
        raise ValueError("weights must be strictly positive (filter zeros upstream)")

    # Each block tracks [weight, weighted_sum, start, end).
    blocks: list[list[float]] = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        blocks.append([wi, wi * yi, i, i + 1])
        # Merge adjacent violators: while the previous block's mean exceeds
        # the current block's mean, pool them (weighted average).
        while len(blocks) >= 2:
            prev, cur = blocks[-2], blocks[-1]
            if prev[1] / prev[0] <= cur[1] / cur[0]:
                break
            merged = [
                prev[0] + cur[0],
                prev[1] + cur[1],
                prev[2],
                cur[3],
            ]
            blocks[-2:] = [merged]

    return PavResult(
        blocks=[
            PavBlock(start=int(b[2]), end=int(b[3]), weight=b[0], value=b[1] / b[0])
            for b in blocks
        ]
    )
