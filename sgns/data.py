"""Raw text validation and deterministic tokenization.

Contract:
    * A corpus is one unicode string. Sentences are separated by newlines.
    * Tokens are maximal runs of unicode letters/digits/underscore and are
      lower-cased. This keeps the synthetic fixtures unambiguous.
    * Empty/whitespace-only corpora and corpora without any token are
      ``InputError``.
    * ``max_tokens`` is an explicit caller-supplied guard; exceeding it is
      ``ResourceExhaustedError`` (a resource limit, not bad input).
"""

from __future__ import annotations

import re

from .errors import InputError, ResourceExhaustedError

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)


def validate_corpus_text(text: object, max_tokens: int | None = None) -> str:
    """Validate a corpus payload and return the normalized unicode string."""
    if not isinstance(text, str):
        raise InputError(
            "corpus must be a string",
            actual_type=type(text).__name__,
        )
    if not text.strip():
        raise InputError("corpus is empty or whitespace only")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise InputError("max_tokens must be a positive integer")
    return text


def tokenize_sentence(sentence: str) -> list[str]:
    """Lower-case a single sentence into its tokens."""
    return [match.group(0).lower() for match in _WORD_RE.finditer(sentence)]


def tokenize_sentences(
    text: str,
    max_tokens: int | None = None,
) -> list[list[str]]:
    """Split newlines into sentences and tokenize each.

    Sentences that contain no token are dropped, but at least one token must
    exist overall.
    """
    sentences: list[list[str]] = []
    total = 0
    for raw_line in text.splitlines():
        tokens = tokenize_sentence(raw_line)
        if not tokens:
            continue
        total += len(tokens)
        if max_tokens is not None and total > max_tokens:
            raise ResourceExhaustedError(
                "corpus exceeds the declared token budget",
                limit=max_tokens,
                observed=total,
            )
        sentences.append(tokens)
    if not sentences:
        raise InputError("corpus contains no recognizable tokens")
    return sentences


def corpus_stats(sentences: list[list[str]]) -> dict[str, int]:
    """Return cheap structural counts used in API responses."""
    return {
        "sentences": len(sentences),
        "tokens": sum(len(s) for s in sentences),
    }
