"""Corpus validation, frozen vocabulary, and deterministic sub-sampling."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError

TOKEN_RE = re.compile(r"^(?![0-9_])\w+$", re.UNICODE)


def validate_corpus(corpus: object) -> list[list[str]]:
    """Validate the API/storage corpus contract and return it unchanged."""
    if not isinstance(corpus, list) or not corpus:
        raise ValidationError(
            "corpus must be a non-empty list of sentences",
            {"type": type(corpus).__name__},
        )
    normalized: list[list[str]] = []
    for sentence_pos, sentence in enumerate(corpus):
        if not isinstance(sentence, list) or not sentence:
            raise ValidationError(
                "each sentence must be a non-empty list of tokens",
                {"sentence_index": sentence_pos},
            )
        for token_pos, token in enumerate(sentence):
            if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
                raise ValidationError(
                    "tokens must be non-empty words containing letters/digits, "
                    "not starting with a digit or underscore",
                    {
                        "sentence_index": sentence_pos,
                        "token_index": token_pos,
                        "token": token if isinstance(token, str) else None,
                    },
                )
        normalized.append(sentence)
    return normalized


@dataclass(frozen=True)
class Vocab:
    """Immutable token-to-row mapping with explicit training distributions."""

    tokens: tuple[str, ...]
    index: dict[str, int]
    counts: np.ndarray
    frequencies: np.ndarray
    negative_distribution: np.ndarray

    @property
    def size(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self.index


def build_vocab(corpus: list[list[str]], *, min_count: int) -> Vocab:
    validate_corpus(corpus)
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValidationError("min_count must be a positive integer")

    counter = Counter(token for sentence in corpus for token in sentence)
    kept = [(token, count) for token, count in counter.items() if count >= min_count]
    kept.sort(key=lambda item: (-item[1], item[0]))
    if not kept:
        raise ValidationError(
            "vocabulary is empty after min_count filtering",
            {"min_count": min_count},
        )

    tokens = tuple(token for token, _ in kept)
    counts = np.asarray([count for _, count in kept], dtype=np.int64)
    frequencies = counts.astype(np.float64) / float(counts.sum())
    weights = counts.astype(np.float64) ** 0.75
    negative_distribution = weights / weights.sum()
    return Vocab(
        tokens=tokens,
        index={token: idx for idx, token in enumerate(tokens)},
        counts=counts,
        frequencies=frequencies,
        negative_distribution=negative_distribution,
    )


def keep_probability(frequency: float, subsample_t: float) -> float:
    """Word2Vec discard threshold's equivalent token keep probability."""
    if frequency <= 0.0:
        return 0.0
    return float(min(1.0, (np.sqrt(frequency / subsample_t) + 1.0) * subsample_t / frequency))


def subsample_sentence(
    sentence: list[str],
    vocab: Vocab,
    rng: np.random.Generator,
    *,
    subsample_t: float,
) -> list[str]:
    """Keep each surviving occurrence independently using one RNG draw."""
    kept: list[str] = []
    for token in sentence:
        token_id = vocab.index.get(token)
        if token_id is None:
            continue
        probability = keep_probability(float(vocab.frequencies[token_id]), subsample_t)
        if float(rng.random()) < probability:
            kept.append(token)
    return kept


def corpus_fingerprint(corpus: list[list[str]]) -> str:
    """Stable SHA-256 over the exact ordered training tokens."""
    payload = json.dumps(corpus, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
