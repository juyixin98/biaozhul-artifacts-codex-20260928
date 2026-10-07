"""Training-pair generation and negative-sampling policy."""
from __future__ import annotations

import numpy as np

from .data import Vocab, subsample_sentence
from .errors import ResourceExhaustedError


def draw_negative_indexes(
    rng: np.random.Generator,
    distribution: np.ndarray,
    *,
    positive_index: int,
    count: int,
    max_attempts: int | None = None,
    on_reject=None,
) -> list[int]:
    """Draw negative indices for one positive target.

    A sample equal to the positive target is rejected and replaced.  Negative
    samples are otherwise independent and may repeat.
    """
    budget = max_attempts if max_attempts is not None else max(32, count * 4)
    accepted: list[int] = []
    attempts = 0
    while len(accepted) < count:
        if attempts >= budget:
            raise ResourceExhaustedError(
                "negative sampling rejected too many colliding samples",
                {
                    "positive_index": positive_index,
                    "needed": count,
                    "accepted": len(accepted),
                    "attempts": attempts,
                    "max_attempts": budget,
                },
            )
        candidate = int(rng.choice(len(distribution), p=distribution))
        attempts += 1
        if candidate != positive_index:
            accepted.append(candidate)
        elif on_reject is not None:
            on_reject(attempts, candidate)
    return accepted


def build_epoch_pairs(
    corpus: list[list[str]],
    vocab: Vocab,
    subsample_rng: np.random.Generator,
    *,
    window: int,
    subsample_t: float,
) -> list[tuple[int, int]]:
    """Build deterministic symmetric center/context pairs after sub-sampling."""
    pairs: list[tuple[int, int]] = []
    for sentence in corpus:
        filtered = subsample_sentence(
            sentence, vocab, subsample_rng, subsample_t=subsample_t
        )
        ids = [vocab.index[token] for token in filtered]
        for center_pos, center_id in enumerate(ids):
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                context_pos = center_pos + offset
                if 0 <= context_pos < len(ids):
                    pairs.append((center_id, ids[context_pos]))
    return pairs
