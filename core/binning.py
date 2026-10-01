"""Deterministic fixed and tie-safe equal-frequency bin boundaries."""

from __future__ import annotations

import numpy as np

from .contracts import BinningMethod
from .errors import BinningError


def fixed_edges(n_bins: int) -> np.ndarray:
    """Return uniform fixed edges, including both probability endpoints."""
    if n_bins < 1:
        raise BinningError("n_bins must be positive")
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=float)


def quantile_edges(p: np.ndarray, w: np.ndarray, n_bins: int) -> np.ndarray:
    """Build weighted equal-frequency edges without splitting tied probabilities.

    Bin cuts are placed only between distinct observed probabilities. For a target
    cumulative mass, the whole tied score containing that target is placed on the
    left. When the requested number of bins exceeds what the distinct scores can
    support, repeated cuts represent explicitly reported empty bins.
    """
    if n_bins < 1:
        raise BinningError("n_bins must be positive")

    values, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    group_weights = np.zeros(values.shape, dtype=float)
    np.add.at(group_weights, inverse, w)

    positive_groups = group_weights > 0.0
    positive_values = values[positive_groups]
    positive_weights = group_weights[positive_groups]
    total_weight = float(np.sum(positive_weights))
    cumulative = np.cumsum(positive_weights)

    cuts: list[float] = []
    previous_group: int | None = None
    for k in range(1, n_bins + 1):
        target = total_weight * k / n_bins
        if k == n_bins:
            cuts.append(1.0)
            continue

        group_at_target = int(np.searchsorted(cumulative, target, side="left"))
        if group_at_target >= len(positive_values) - 1 or group_at_target == previous_group:
            # No distinct higher score exists; retain this requested slot as empty.
            cuts.append(1.0)
        else:
            left_score = float(positive_values[group_at_target])
            right_score = float(positive_values[group_at_target + 1])
            cuts.append((left_score + right_score) / 2.0)
            previous_group = group_at_target

    return np.asarray([0.0, *cuts], dtype=float)


def resolve_edges(
    p: np.ndarray,
    w: np.ndarray,
    method: BinningMethod,
    n_bins: int,
    explicit_edges: list[float] | None,
) -> np.ndarray:
    if method == "fixed":
        if explicit_edges is not None:
            return np.asarray(explicit_edges, dtype=float)
        return fixed_edges(n_bins)
    if explicit_edges is not None:
        return np.asarray(explicit_edges, dtype=float)
    return quantile_edges(p, w, n_bins)


def assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign probabilities to [lower, upper) bins; the final bin includes 1.0."""
    cuts = edges[1:-1]
    return np.searchsorted(cuts, p, side="right").astype(int)
