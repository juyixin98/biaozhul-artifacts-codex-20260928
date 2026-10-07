"""Negative sampling distribution and the positive-collision rule.

The negative distribution is fixed *before* training starts:

    P_neg(w) = count(w) ** exponent / sum_u count(u) ** exponent,

with ``exponent = 0.75`` by default. A cumulative-probability table is used
so each draw is a single inverse-CDF lookup against a fresh uniform number
from the replayable RNG.

Collision handling is fixed: a sampled negative equal to the positive target
is rejected and immediately redrawn with another fresh uniform draw, using
the same distribution. Duplicate negatives inside one target are allowed
(only the positive target is forbidden). If ``max_attempts`` draws all
collide the kernel raises ``ComputationFailureError``; this is a failure to
produce a valid sample, not an input or resource error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ComputationFailureError
from .vocab import Vocabulary


@dataclass(frozen=True)
class NegativeTable:
    """Cumulative distribution used for inverse-CDF negative draws."""

    probabilities: tuple[float, ...]
    cumulative: tuple[float, ...]
    exponent: float

    @classmethod
    def build(
        cls,
        vocab: Vocabulary,
        exponent: float = 0.75,
    ) -> "NegativeTable":
        weights = np.asarray(vocab.counts, dtype=np.float64) ** float(exponent)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ComputationFailureError(
                "negative sampling weights are not a finite positive distribution",
                weight_sum=total,
            )
        probs = weights / total
        cumulative = tuple(np.cumsum(probs).tolist())
        # Guarantee the final boundary is exactly 1 despite float rounding.
        cumulative_list = list(cumulative)
        cumulative_list[-1] = 1.0
        return cls(
            probabilities=tuple(probs.tolist()),
            cumulative=tuple(cumulative_list),
            exponent=float(exponent),
        )

    def draw_one(self, rng: np.random.Generator) -> int:
        """Draw one word id without any collision constraint."""
        u = float(rng.random(dtype=np.float64))
        # searchsorted on the right gives the inverse CDF; u==0 selects the
        # first bucket because cumulative is strictly increasing.
        return int(np.searchsorted(np.asarray(self.cumulative), u, side="right"))

    def sample_negatives(
        self,
        positive_id: int,
        count: int,
        rng: np.random.Generator,
        max_attempts: int = 100,
    ) -> list[int]:
        negatives, _ = self.sample_negatives_tracked(
            positive_id, count, rng, max_attempts=max_attempts
        )
        return negatives

    def sample_negatives_tracked(
        self,
        positive_id: int,
        count: int,
        rng: np.random.Generator,
        max_attempts: int = 100,
    ) -> tuple[list[int], int]:
        """Draw ``count`` negative ids, redrawing positive-target collisions."""
        if count <= 0:
            raise ValueError("count must be positive")
        negatives: list[int] = []
        redraws = 0
        draws = 0
        while len(negatives) < count:
            draws += 1
            if draws > max_attempts:
                raise ComputationFailureError(
                    "negative sampling exhausted its redraw budget while "
                    "avoiding the positive target",
                    positive_id=positive_id,
                    attempts=redraws,
                    required=count,
                )
            candidate = self.draw_one(rng)
            if candidate == positive_id:
                redraws += 1
                continue
            negatives.append(candidate)
        return negatives, redraws
