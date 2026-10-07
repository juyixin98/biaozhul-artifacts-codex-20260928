"""Vocabulary construction, freezing, and word2vec-style subsampling.

The vocabulary is built once from the corpus and then *frozen*: training,
querying and persistence all share the same word-to-id table. Words below
``min_count`` are dropped before freezing, but their raw counts are retained
so tests can assert the exact cutoff decision.

Subsampling uses the explicit word2vec formula

    keep(w) = min(1.0, sqrt(freq(w) / t) + freq(w) / t)       (t > 0)
    keep(w) = 1.0                                             (t = 0)

where ``freq(w) = count(w) / total_tokens``. A token is kept when a fresh
uniform draw from the replayable RNG is strictly below ``keep(w)``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .errors import InputError


@dataclass(frozen=True)
class Vocabulary:
    """Frozen vocabulary plus the statistics subsampling needs."""

    index_to_word: tuple[str, ...]
    _word_to_index: dict[str, int]
    counts: tuple[int, ...]
    raw_counts: dict[str, int]
    total_tokens: int
    min_count: int
    subsample_t: float

    def __post_init__(self) -> None:
        if not self.index_to_word:
            raise InputError("cannot freeze an empty vocabulary")

    @property
    def size(self) -> int:
        return len(self.index_to_word)

    def to_dict(self) -> dict[str, object]:
        return {
            "index_to_word": list(self.index_to_word),
            "counts": list(self.counts),
            "raw_counts": dict(self.raw_counts),
            "total_tokens": self.total_tokens,
            "min_count": self.min_count,
            "subsample_t": self.subsample_t,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "Vocabulary":
        return cls(
            index_to_word=tuple(raw["index_to_word"]),  # type: ignore[arg-type]
            _word_to_index={
                word: i for i, word in enumerate(raw["index_to_word"])  # type: ignore[arg-type]
            },
            counts=tuple(raw["counts"]),  # type: ignore[arg-type]
            raw_counts=dict(raw["raw_counts"]),  # type: ignore[arg-type]
            total_tokens=int(raw["total_tokens"]),  # type: ignore[arg-type]
            min_count=int(raw["min_count"]),  # type: ignore[arg-type]
            subsample_t=float(raw["subsample_t"]),  # type: ignore[arg-type]
        )

    def index(self, word: str) -> int:
        """Return the frozen id for ``word`` or raise ``KeyError``."""
        try:
            return self.word_to_index[word]
        except KeyError:
            raise

    def contains(self, word: str) -> bool:
        return word in self.word_to_index

    @property
    def word_to_index(self) -> dict[str, int]:
        return self._word_to_index

    def frequencies(self) -> np.ndarray:
        return np.asarray(self.counts, dtype=np.float64) / float(self.total_tokens)

    def keep_probabilities(self) -> np.ndarray:
        """Explicit subsampling keep probabilities for every frozen word."""
        if self.subsample_t <= 0.0:
            return np.ones(self.size, dtype=np.float64)
        freq = self.frequencies()
        ratio = freq / self.subsample_t
        return np.minimum(1.0, np.sqrt(ratio) + ratio)


def build_vocabulary(
    sentences: list[list[str]],
    min_count: int = 1,
    subsample_t: float = 1e-3,
) -> Vocabulary:
    """Count tokens, apply ``min_count`` and freeze the resulting vocabulary.

    Tie breaking is deterministic: retained words are ordered by descending
    count, then ascending lexicographic word.
    """
    if not isinstance(min_count, int) or min_count <= 0:
        raise InputError("min_count must be a positive integer")
    if not 0.0 <= subsample_t <= 1.0:
        raise InputError("subsample_t must be in [0.0, 1.0]")
    counter: Counter[str] = Counter()
    total = 0
    for sentence in sentences:
        counter.update(sentence)
        total += len(sentence)
    if total == 0:
        raise InputError("cannot build vocabulary from zero tokens")
    retained = [
        (word, count)
        for word, count in counter.items()
        if count >= min_count
    ]
    retained.sort(key=lambda item: (-item[1], item[0]))
    words = tuple(word for word, _ in retained)
    counts = tuple(count for _, count in retained)
    return Vocabulary(
        index_to_word=words,
        _word_to_index={word: i for i, word in enumerate(words)},
        counts=counts,
        raw_counts=dict(counter),
        total_tokens=total,
        min_count=min_count,
        subsample_t=subsample_t,
    )


def encode_sentence(sentence: list[str], vocab: Vocabulary) -> list[int]:
    """Map tokens to frozen ids, silently dropping out-of-vocabulary tokens.

    Dropped tokens are words pruned by ``min_count``; the vocabulary itself
    never changes during training.
    """
    table = vocab.word_to_index
    return [table[token] for token in sentence if token in table]


def apply_subsampling(
    token_ids: list[int],
    vocab: Vocabulary,
    rng: np.random.Generator,
) -> list[int]:
    """Return the tokens surviving one replayable subsampling pass."""
    keep = vocab.keep_probabilities()
    result: list[int] = []
    for token_id in token_ids:
        if rng.random(dtype=np.float64) < keep[token_id]:
            result.append(token_id)
    return result
