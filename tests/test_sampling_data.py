"""Tests for validation, vocabulary freezing, subsampling and negatives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sgns.data import tokenize_sentences, validate_corpus_text
from sgns.errors import (
    ComputationFailureError,
    InputError,
    ResourceExhaustedError,
)
from sgns.sampling import NegativeTable
from sgns.vocab import apply_subsampling, build_vocabulary

from .replay_log import RunLogger


CORPUS = "the quick brown fox\n" "the dog the fox quick\n"


def test_tokenization_and_validation_categories() -> None:
    log = RunLogger("validation_categories")
    with pytest.raises(InputError) as empty:
        validate_corpus_text("   \n\t")
    assert empty.value.code == "INPUT_ERROR"
    with pytest.raises(InputError) as wrong_type:
        validate_corpus_text(["not a string"])  # type: ignore[arg-type]
    assert wrong_type.value.code == "INPUT_ERROR"
    # Token-level budget is enforced during tokenization with observed count.
    with pytest.raises(ResourceExhaustedError) as exhausted:
        tokenize_sentences("a b c", max_tokens=2)
    assert exhausted.value.code == "RESOURCE_EXHAUSTED"
    assert exhausted.value.details == {"limit": 2, "observed": 3}
    sentences = tokenize_sentences(CORPUS)
    assert sentences == [
        ["the", "quick", "brown", "fox"],
        ["the", "dog", "the", "fox", "quick"],
    ]
    log.record(
        "input_categories_distinct",
        "empty/type problems are INPUT_ERROR; declared-budget overflow is "
        "RESOURCE_EXHAUSTED; tokenization is newline and lowercase based",
        expected={
            "empty": "INPUT_ERROR",
            "type": "INPUT_ERROR",
            "budget": "RESOURCE_EXHAUSTED",
            "token_count": 9,
        },
        observed={
            "empty": empty.value.code,
            "type": wrong_type.value.code,
            "budget": exhausted.value.code,
            "tokens": [t for s in sentences for t in s],
        },
    )


def test_vocabulary_freeze_and_min_count() -> None:
    log = RunLogger("vocabulary_freeze")
    sentences = tokenize_sentences(CORPUS)
    vocab = build_vocabulary(sentences, min_count=2, subsample_t=0.0)
    # "brown" and "dog" occur once and must be pruned; raw counts retained.
    assert vocab.index_to_word == ("the", "fox", "quick")
    assert vocab.counts == (3, 2, 2)
    assert vocab.raw_counts["dog"] == 1
    assert vocab.raw_counts["brown"] == 1
    # Frozen lookups use the same ids for training and querying.
    assert vocab.index("fox") == 1
    with pytest.raises(KeyError):
        vocab.index("dog")
    assert vocab.size == 3
    log.record(
        "min_count_frozen",
        "min_count=2 freezes only types with count>=2, ordered by descending "
        "count then lexicographic; pruned raw counts remain auditable",
        expected={"frozen": ["the", "fox", "quick"], "counts": [3, 2, 2]},
        observed={
            "frozen": list(vocab.index_to_word),
            "counts": list(vocab.counts),
            "raw_dog": vocab.raw_counts["dog"],
        },
    )


def test_negative_distribution_is_unigram_three_quarters() -> None:
    log = RunLogger("negative_distribution")
    vocab = build_vocabulary(
        tokenize_sentences("a a a a a a a b b b c"),
        subsample_t=0.0,
    )
    table = NegativeTable.build(vocab, exponent=0.75)
    weights = [7**0.75, 3**0.75, 1**0.75]
    total = sum(weights)
    expected = [w / total for w in weights]
    assert list(table.probabilities) == pytest.approx(expected, abs=1e-12)
    # 60,000 empirical draws from a fixed seed: frequencies track the table.
    rng = np.random.default_rng(2026)
    counts = [0, 0, 0]
    draws = 60_000
    for _ in range(draws):
        counts[table.draw_one(rng)] += 1
    observed = [c / draws for c in counts]
    assert observed == pytest.approx(expected, abs=0.01)
    log.record(
        "inverse_cdf_frequency",
        "P(w)=count^0.75/sum; 60k fixed-seed draws must approximate the exact "
        "distribution within a 1% tolerance",
        expected=expected,
        observed=observed,
        seed=2026,
        draws=draws,
    )


def test_positive_collision_rule_redraws_and_is_replayable() -> None:
    log = RunLogger("positive_collision")
    vocab = build_vocabulary([["a", "b"]], subsample_t=0.0)
    table = NegativeTable.build(vocab, exponent=0.0)  # uniform 0.5/0.5
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    # Positive id 0; returned negatives must never equal 0.
    negatives, redraws = table.sample_negatives_tracked(0, 4, rng_a)
    assert all(n != 0 for n in negatives)
    assert negatives == [1, 1, 1, 1]
    assert redraws >= 1
    # Consume the same raw stream independently; replay matches.
    replay, replay_redraws = table.sample_negatives_tracked(0, 4, rng_b)
    assert replay == negatives
    assert replay_redraws == redraws
    log.record(
        "positive_never_negative",
        "with two equiprobable words, positives are rejected and redrawn; "
        "replaying the same RNG seed yields identical negatives/redraw counts",
        expected={"all_negatives": [1, 1, 1, 1]},
        observed={
            "negatives": negatives,
            "redraws": redraws,
            "replay_matches": replay == negatives and replay_redraws == redraws,
        },
        seed=42,
    )


def test_negative_redraw_budget_failure_category() -> None:
    log = RunLogger("negative_budget")
    vocab = build_vocabulary([["a", "b"]], subsample_t=0.0)
    table = NegativeTable.build(vocab, exponent=0.0)

    class AlwaysPositive:
        def random(self, dtype=np.float64) -> float:  # noqa: D401
            return 0.0  # always lands in first CDF bucket -> positive

    with pytest.raises(ComputationFailureError) as exc:
        table.sample_negatives_tracked(
            0,
            1,
            AlwaysPositive(),  # type: ignore[arg-type]
            max_attempts=5,
        )
    assert exc.value.code == "COMPUTATION_FAILED"
    assert exc.value.details["attempts"] == 5
    log.record(
        "redraw_budget_category",
        "inability to produce a valid negative sample is COMPUTATION_FAILED "
        "and records the exhausted attempt count",
        expected={"code": "COMPUTATION_FAILED", "attempts": 5},
        observed={"code": exc.value.code, "attempts": exc.value.details["attempts"]},
    )


def test_subsampling_probabilities_and_deterministic_keeps() -> None:
    log = RunLogger("subsampling")
    # 100,000 tokens: "a" 90,000, "b" 9,999, "rare" once; t = 1e-3.
    sentences = [["a"] * 90_000 + ["b"] * 9_999 + ["rare"]]
    vocab = build_vocabulary(sentences, subsample_t=1e-3)
    keep = vocab.keep_probabilities()
    total = 100_000
    rare_freq = 1 / total
    uncapped_rare = math.sqrt(rare_freq / 1e-3) + rare_freq / 1e-3
    assert keep[0] == 1.0 and keep[1] == 1.0
    assert keep[2] == pytest.approx(uncapped_rare, abs=1e-12)
    assert uncapped_rare == pytest.approx(0.11, abs=1e-9)
    assert 0.0 < keep[2] < 0.12

    # Empirical subsampling of the rare word uses the explicit probability
    # (a single corpus token is exercised repeatedly at the fixed RNG seed).
    rng_freq = np.random.default_rng(77)
    rare_id = vocab.index("rare")
    draws = 40_000
    rare_tokens = [rare_id] * draws
    kept_rare = apply_subsampling(rare_tokens, vocab, rng_freq)
    observed_fraction = len(kept_rare) / draws
    assert observed_fraction == pytest.approx(uncapped_rare, abs=0.01)

    rng = np.random.default_rng(8)
    ids = [vocab.index("a"), vocab.index("b")]
    kept_a = apply_subsampling([ids[0]] * 200, vocab, rng)
    rng_a = np.random.default_rng(8)
    rng_b = np.random.default_rng(8)
    assert apply_subsampling([ids[0]] * 200, vocab, rng_a) == kept_a
    # t=0 disables subsampling entirely.
    vocab_no_sub = build_vocabulary(sentences, subsample_t=0.0)
    assert apply_subsampling([0, 1, 0, 1], vocab_no_sub, rng_b) == [0, 1, 0, 1]
    log.record(
        "subsampling_explicit",
        "keep(w)=min(1,sqrt(f/t)+f/t); a at f=0.9 is always kept, b at "
        "rare f=1e-5, t=1e-3 gives sqrt(0.01)+0.01=0.11; fixed-seed draws "
        "replay exactly and the empirical fraction tracks the probability",
        expected={"keep_frequent": 1.0, "keep_rare": 0.11},
        observed={
            "keep_a": float(keep[0]),
            "keep_rare_exact": float(keep[2]),
            "keep_rare_empirical": observed_fraction,
        },
        seed=(8, 77),
    )
