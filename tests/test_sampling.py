"""Small-vocabulary sampling frequencies and fixed collision policy."""
from __future__ import annotations

import numpy as np
import pytest

from app.data import build_vocab, keep_probability
from app.errors import ResourceExhaustedError
from app.sampling import build_epoch_pairs, draw_negative_indexes
from .reference import reference_keep_probability


class ScriptedRandom:
    def __init__(self, values: list[float]):
        self.values = values
        self.calls: list[float] = []

    def random(self) -> float:
        value = self.values.pop(0)
        self.calls.append(value)
        return value


class ScriptedChoice:
    def __init__(self, choices: list[int]):
        self.choices = choices
        self.calls = 0

    def choice(self, size: int, p: np.ndarray) -> int:
        assert size == len(p)
        value = self.choices.pop(0)
        self.calls += 1
        return value


def test_vocab_frequency_and_unigram_0_75_negative_distribution(evidence):
    corpus = [["cat"] * 10 + ["dog"] * 5]
    vocab = build_vocab(corpus, min_count=1)

    assert vocab.tokens == ("cat", "dog")
    assert vocab.counts.tolist() == [10, 5]
    assert vocab.frequencies == pytest.approx([2 / 3, 1 / 3])
    cat_weight = 10.0**0.75
    dog_weight = 5.0**0.75
    expected = [cat_weight / (cat_weight + dog_weight), dog_weight / (cat_weight + dog_weight)]
    assert vocab.negative_distribution == pytest.approx(expected)
    assert abs(sum(vocab.negative_distribution) - 1.0) < 1e-15
    evidence(
        "negative_distribution",
        {"formula": "count^0.75/sum(count^0.75)", "values": expected},
        vocab.negative_distribution.tolist(),
        "negative sampling must use an explicitly normalized unigram^0.75 table",
    )


def test_subsampling_keep_probability_and_independent_token_draws(evidence):
    corpus = [["cat", "dog", "cat"]]
    vocab = build_vocab(corpus, min_count=1)
    threshold = 0.1
    cat_id, dog_id = 0, 1
    expected_cat = reference_keep_probability(2 / 3, threshold)
    expected_dog = reference_keep_probability(1 / 3, threshold)

    assert keep_probability(float(vocab.frequencies[cat_id]), threshold) == pytest.approx(
        expected_cat
    )
    assert keep_probability(float(vocab.frequencies[dog_id]), threshold) == pytest.approx(
        expected_dog
    )
    rng = ScriptedRandom([0.6, 0.2, 0.6])
    from app.data import subsample_sentence

    result = subsample_sentence(
        ["cat", "dog", "cat"], vocab, rng, subsample_t=threshold
    )
    assert result == ["dog"]
    assert rng.calls == [0.6, 0.2, 0.6]
    evidence(
        "subsample_draws",
        {"cat_probability": expected_cat, "dog_probability": expected_dog, "kept": ["dog"]},
        {"probabilities": [expected_cat, expected_dog], "result": result, "draws": rng.calls},
        "each occurrence independently compares one uniform draw with its keep probability",
    )


def test_colliding_negative_is_redrawn_but_repeated_negatives_remain(evidence):
    distribution = np.array([0.7, 0.3])
    rng = ScriptedChoice([0, 1, 1])
    negatives = draw_negative_indexes(
        rng, distribution, positive_index=0, count=2, max_attempts=10
    )
    assert negatives == [1, 1]
    assert rng.calls == 3
    evidence(
        "collision_rejection",
        {"accepted": [1, 1], "attempts": 3},
        {"accepted": negatives, "attempts": rng.calls},
        "only positive equality is rejected; independent negatives may repeat",
    )


def test_negative_budget_exhaustion_has_resource_category():
    rng = ScriptedChoice([0, 0])
    with pytest.raises(ResourceExhaustedError) as excinfo:
        draw_negative_indexes(
            rng,
            np.array([0.7, 0.3]),
            positive_index=0,
            count=1,
            max_attempts=2,
        )
    assert excinfo.value.category == "resource_exhausted"
    assert excinfo.value.details["attempts"] == 2


def test_epoch_pair_order_is_explicit_and_window_bounded():
    vocab = build_vocab([["a", "b", "a"]], min_count=1)
    pairs = build_epoch_pairs(
        [["a", "b", "a"]],
        vocab,
        ScriptedRandom([0.5, 0.5, 0.5]),
        window=1,
        subsample_t=0.9,
    )
    assert pairs == [(0, 1), (1, 0), (1, 0), (0, 1)]
