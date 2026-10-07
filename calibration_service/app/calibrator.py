"""Isotonic probability calibrator: fit via PAV, predict as a step function."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pav import pav
from .validation import (
    ValidationSummary,
    merge_duplicate_scores,
    validate_arrays,
)

# Out-of-range scores are clamped to the nearest boundary probability
# (constant extrapolation). This is the only supported policy.
EXTRAPOLATION_POLICY = "clamp_to_boundary"


@dataclass
class IsotonicCalibrator:
    """Piecewise-constant, non-decreasing score -> probability mapping."""

    score_field: str  # meaning of the input score, bound at fit time
    thresholds: np.ndarray = field(default_factory=lambda: np.empty(0))
    probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))
    extrapolation: str = EXTRAPOLATION_POLICY
    validation_summary: ValidationSummary | None = None

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        score_field: str,
    ) -> "IsotonicCalibrator":
        s, y, w, summary = validate_arrays(scores, labels, weights)
        s, y, w, n_merged = merge_duplicate_scores(s, y, w)
        summary.n_duplicate_scores_merged = n_merged
        if n_merged:
            summary.warnings.append(
                f"merged {n_merged} rows into duplicate-score groups"
            )
        result = pav(y, w)
        # Block k covers scores up to its right edge (largest score in the
        # block). Predict uses searchsorted on these edges.
        edges = np.array([s[b.end - 1] for b in result.blocks], dtype=float)
        probs = np.array([b.value for b in result.blocks], dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        return cls(
            score_field=score_field,
            thresholds=edges,
            probabilities=probs,
            validation_summary=summary,
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Map scores to probabilities.

        Step function: a score falls into the first block whose right edge
        is >= the score. Scores below the training range get the first
        block's probability, scores above get the last block's (clamp).
        """
        if self.thresholds.size == 0:
            raise RuntimeError("calibrator is not fitted")
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite")
        idx = np.searchsorted(self.thresholds, scores, side="left")
        idx = np.clip(idx, 0, self.probabilities.size - 1)
        return self.probabilities[idx]

    def is_monotone(self) -> bool:
        return bool(np.all(np.diff(self.probabilities) >= 0))

    def to_dict(self) -> dict:
        return {
            "score_field": self.score_field,
            "thresholds": self.thresholds.tolist(),
            "probabilities": self.probabilities.tolist(),
            "extrapolation": self.extrapolation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        return cls(
            score_field=d["score_field"],
            thresholds=np.asarray(d["thresholds"], dtype=float),
            probabilities=np.asarray(d["probabilities"], dtype=float),
            extrapolation=d.get("extrapolation", EXTRAPOLATION_POLICY),
        )
