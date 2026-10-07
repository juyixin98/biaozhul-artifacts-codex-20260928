"""Kernel correctness tests with independently hand-computed expectations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sgns.errors import ComputationFailureError
from sgns.kernel import (
    initialize_matrices,
    skipgram_event,
    stable_sigmoid,
    train_skipgram,
)
from sgns.sampling import NegativeTable
from sgns.vocab import build_vocabulary

from . import reference
from .replay_log import RunLogger


def test_stable_sigmoid_hand_values_and_extremes() -> None:
    log = RunLogger("sigmoid_extremes")
    cases = [
        (0.0, 0.5),
        (0.06, 0.51499550161941),
        (-0.14, 0.4650570548417855),
        (0.38, 0.5938731029341427),
        (700.0, 1.0),
        (-700.0, 0.0),
    ]
    for x, expected in cases:
        observed = stable_sigmoid(x)
        assert math.isfinite(observed)
        assert observed == pytest.approx(expected, abs=1e-12, rel=0)
        log.record(
            "stable_sigmoid",
            "sigmoid must stay finite at +/-700 and match scalar reference",
            expected=expected,
            observed=observed,
            x=x,
        )


def test_single_event_matches_hand_computed_oracle() -> None:
    log = RunLogger("single_event_oracle")
    # Vocabulary ids: 0=center c, 1=positive p, 2=n1, 3=n2.
    vocab = build_vocabulary(
        [["c", "p", "n1", "n2"]],
        min_count=1,
        subsample_t=0.0,
    )
    input_vectors = np.array([reference.HAND_U, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
    output_vectors = np.array(
        [
            [0.0, 0.0],
            reference.HAND_VP,
            reference.HAND_VN1,
            reference.HAND_VN2,
        ],
        dtype=float,
    )
    scores, coefficients = skipgram_event(
        input_vectors,
        output_vectors,
        center_id=0,
        positive_id=1,
        negative_ids=[2, 3],
        lr=reference.HAND_LR,
    )
    assert scores == pytest.approx(reference.HAND_SCORES, abs=1e-12)
    assert coefficients == pytest.approx(reference.HAND_COEFFICIENTS, abs=1e-12)
    assert input_vectors[0].tolist() == pytest.approx(reference.HAND_U_AFTER, abs=1e-12)
    assert output_vectors[1].tolist() == pytest.approx(reference.HAND_VP_AFTER, abs=1e-12)
    assert output_vectors[2].tolist() == pytest.approx(reference.HAND_VN1_AFTER, abs=1e-12)
    assert output_vectors[3].tolist() == pytest.approx(reference.HAND_VN2_AFTER, abs=1e-12)
    # Unused input rows must be untouched: the two vector tables are separate.
    assert input_vectors[1].tolist() == [0.0, 0.0]
    assert output_vectors[0].tolist() == [0.0, 0.0]
    log.record(
        "event_matches_oracle",
        "scores, both signed coefficients, input update and each distinct "
        "output row match the independent hand oracle; no cross-table overwrite",
        expected={
            "scores": reference.HAND_SCORES,
            "coefficients": reference.HAND_COEFFICIENTS,
            "u_after": reference.HAND_U_AFTER,
            "vp_after": reference.HAND_VP_AFTER,
        },
        observed={
            "scores": scores,
            "coefficients": coefficients,
            "u_after": input_vectors[0].tolist(),
            "vp_after": output_vectors[1].tolist(),
        },
    )


def test_duplicate_negative_accumulates_on_same_preupdate_row() -> None:
    log = RunLogger("duplicate_negative")
    u_before = reference.HAND_U
    rows = {
        1: list(reference.HAND_VP),
        2: list(reference.HAND_VN1),
    }
    scores_ref, coef_ref, u_ref, v_ref = reference.ref_event(
        u_before,
        rows,
        positive_id=1,
        negative_ids=[2, 2],
        lr=reference.HAND_LR,
    )
    input_vectors = np.array([u_before, [0.0, 0.0], [0.0, 0.0]], dtype=float)
    output_vectors = np.array(
        [[0.0, 0.0], reference.HAND_VP, reference.HAND_VN1],
        dtype=float,
    )
    scores, coefficients = skipgram_event(
        input_vectors,
        output_vectors,
        center_id=0,
        positive_id=1,
        negative_ids=[2, 2],
        lr=reference.HAND_LR,
    )
    assert scores == pytest.approx(scores_ref, abs=1e-12)
    assert coefficients == pytest.approx(coef_ref, abs=1e-12)
    assert input_vectors[0].tolist() == pytest.approx(u_ref, abs=1e-12)
    assert output_vectors[2].tolist() == pytest.approx(v_ref[2], abs=1e-12)
    assert output_vectors[1].tolist() == pytest.approx(v_ref[1], abs=1e-12)
    log.record(
        "duplicate_negative",
        "a repeated negative contributes twice against one pre-update row; "
        "the second use must not read the partially updated output vector",
        expected={"v_n1_after": v_ref[2], "u_after": u_ref},
        observed={
            "v_n1_after": output_vectors[2].tolist(),
            "u_after": input_vectors[0].tolist(),
        },
    )


def test_initialization_does_not_overwrite_output_vectors() -> None:
    log = RunLogger("init_separate_tables")
    rng = np.random.default_rng(3)
    input_vectors, output_vectors = initialize_matrices(4, 5, rng)
    assert input_vectors.shape == (4, 5)
    assert output_vectors.shape == (4, 5)
    assert np.count_nonzero(output_vectors) == 0
    assert abs(input_vectors).max() <= 0.5 / 5
    log.record(
        "separate_initialization",
        "input vectors are small random values while output vectors are zero",
        expected={"output_nonzero": 0, "max_abs_input_bound": 0.1},
        observed={
            "output_nonzero": int(np.count_nonzero(output_vectors)),
            "max_abs_input": float(abs(input_vectors).max()),
        },
    )


def test_nonfinite_score_is_computation_failure() -> None:
    log = RunLogger("nonfinite_failure")
    input_vectors = np.array([[float("inf"), 0.0], [0.0, 0.0]], dtype=float)
    output_vectors = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    with pytest.raises(ComputationFailureError) as exc_info:
        skipgram_event(
            input_vectors,
            output_vectors,
            center_id=0,
            positive_id=1,
            negative_ids=[0],
            lr=0.1,
        )
    assert exc_info.value.code == "COMPUTATION_FAILED"
    log.record(
        "nonfinite_classified",
        "inf dot zero produces NaN and must be COMPUTATION_FAILED, not a "
        "silent NaN vector or a generic input error",
        expected="COMPUTATION_FAILED",
        observed=exc_info.value.code,
        details=exc_info.value.details,
    )


def test_training_is_replayable_with_seed() -> None:
    log = RunLogger("kernel_replay")
    sentences = [
        ["a", "b", "a", "b", "c"],
        ["c", "a", "b", "c", "a"],
    ]
    vocab = build_vocabulary(sentences, min_count=1, subsample_t=0.0)
    table = NegativeTable.build(vocab, 0.75)

    from sgns.config import TrainConfig

    config = TrainConfig(
        dim=4,
        lr=0.05,
        epochs=3,
        window=2,
        negatives=2,
        subsample_t=0.0,
        seed=99,
    )
    first = train_skipgram(
        [[vocab.index(t) for t in s] for s in sentences],
        vocab,
        table,
        config,
        record_events=2,
    )
    second = train_skipgram(
        [[vocab.index(t) for t in s] for s in sentences],
        vocab,
        table,
        config,
    )
    assert np.array_equal(first.input_vectors, second.input_vectors)
    assert np.array_equal(first.output_vectors, second.output_vectors)
    assert first.stats.to_dict() == second.stats.to_dict()
    log.record(
        "seeded_replay_identical",
        "two runs with the same seed and identical inputs reproduce every "
        "vector and statistic bit-for-bit",
        expected="identical arrays and stats",
        observed={
            "input_equal": bool(
                np.array_equal(first.input_vectors, second.input_vectors)
            ),
            "output_equal": bool(
                np.array_equal(first.output_vectors, second.output_vectors)
            ),
            "stats": first.stats.to_dict(),
        },
    )
