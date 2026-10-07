"""Input validation and canonicalization for calibration datasets.

Explicit rules (documented in README):
  - score must be finite; NaN/inf score -> row rejected with ValidationError
  - label must be exactly 0 or 1; anything else -> ValidationError
  - weight must be finite and >= 0; negative -> ValidationError
  - weight == 0 rows are dropped (reported in the summary, not an error)
  - duplicate scores are merged by summing weights and taking the
    weight-averaged label (handled in `merge_duplicate_scores`)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ValidationError(ValueError):
    """Raised when a dataset row violates a hard rule."""

    def __init__(self, message: str, row_indices: list[int] | None = None):
        super().__init__(message)
        self.row_indices = row_indices or []


@dataclass
class ValidationSummary:
    n_input: int
    n_used: int
    n_zero_weight_dropped: int = 0
    n_duplicate_scores_merged: int = 0
    warnings: list[str] = field(default_factory=list)


def validate_arrays(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ValidationSummary]:
    """Validate raw arrays; return filtered (score, label, weight) + summary."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = scores.shape[0]
    if not (labels.shape[0] == weights.shape[0] == n):
        raise ValidationError("scores, labels, weights must have equal length")
    if n == 0:
        raise ValidationError("dataset is empty")

    bad_score = np.where(~np.isfinite(scores))[0].tolist()
    if bad_score:
        raise ValidationError(
            f"non-finite score at rows {bad_score[:10]}", bad_score
        )
    bad_label = np.where(~np.isin(labels, (0.0, 1.0)))[0].tolist()
    if bad_label:
        raise ValidationError(
            f"label not in (0, 1) at rows {bad_label[:10]}", bad_label
        )
    bad_weight = np.where(~np.isfinite(weights) | (weights < 0))[0].tolist()
    if bad_weight:
        raise ValidationError(
            f"negative or non-finite weight at rows {bad_weight[:10]}", bad_weight
        )

    keep = weights > 0
    dropped = int((~keep).sum())
    summary = ValidationSummary(
        n_input=n, n_used=int(keep.sum()), n_zero_weight_dropped=dropped
    )
    if dropped:
        summary.warnings.append(f"dropped {dropped} zero-weight rows")
    if summary.n_used == 0:
        raise ValidationError("all rows have zero weight; nothing to train on")
    return scores[keep], labels[keep], weights[keep], summary


def merge_duplicate_scores(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sort by score and merge equal scores into one weighted observation.

    Returns sorted unique scores, merged labels (weighted mean), merged
    weights (sum), and the number of rows removed by merging.
    """
    order = np.argsort(scores, kind="stable")
    s, y, w = scores[order], labels[order], weights[order]
    uniq, starts = np.unique(s, return_index=True)
    if uniq.shape[0] == s.shape[0]:
        return s, y, w, 0
    merged_w = np.add.reduceat(w, starts)
    merged_y = np.add.reduceat(w * y, starts) / merged_w
    return uniq, merged_y, merged_w, int(s.shape[0] - uniq.shape[0])
