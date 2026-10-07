"""Hand-computed one-step SGNS gradient and non-overwrite tests."""
from __future__ import annotations

import math

import numpy as np
import pytest


def approx(values, **kwargs):
    return pytest.approx(values, rel=1e-12, abs=kwargs.get("abs", 1e-12))

from app.model import negative_sampling_step, stable_sigmoid
from .reference import (
    LEARNING_RATE,
    REFERENCE_CENTER,
    REFERENCE_NEGATIVE,
    REFERENCE_POSITIVE,
    dot,
    reference_step,
    sigmoid,
)


def test_hand_computed_single_update_matches_independent_reference(evidence):
    inputs = np.zeros((2, 3), dtype=np.float64)
    outputs = np.zeros((2, 3), dtype=np.float64)
    inputs[0] = REFERENCE_CENTER
    outputs[0] = REFERENCE_POSITIVE
    outputs[1] = REFERENCE_NEGATIVE

    detail = negative_sampling_step(
        inputs,
        outputs,
        center_id=0,
        positive_id=0,
        negative_ids=[1],
        learning_rate=LEARNING_RATE,
    )

    positive = reference_step(
        REFERENCE_CENTER, REFERENCE_POSITIVE, 1.0, LEARNING_RATE
    )
    negative = reference_step(
        REFERENCE_CENTER, REFERENCE_NEGATIVE, 0.0, LEARNING_RATE
    )

    assert detail["scores"] == approx([positive["score"], negative["score"]])
    assert detail["probabilities"] == approx(
        [positive["probability"], negative["probability"]]
    )
    expected_loss = -math.log(positive["probability"]) - math.log(
        1.0 - negative["probability"]
    )
    assert detail["loss"] == approx(expected_loss)

    expected_input_gradient = [
        pos_g + neg_g
        for pos_g, neg_g in zip(
            positive["input_gradient"], negative["input_gradient"]
        )
    ]
    expected_input = [
        c + g for c, g in zip(REFERENCE_CENTER, expected_input_gradient)
    ]
    expected_positive_output = [
        o + g for o, g in zip(REFERENCE_POSITIVE, positive["output_gradient"])
    ]
    expected_negative_output = [
        o + g for o, g in zip(REFERENCE_NEGATIVE, negative["output_gradient"])
    ]

    assert inputs[0].tolist() == approx(expected_input)
    assert outputs[0].tolist() == approx(expected_positive_output)
    assert outputs[1].tolist() == approx(expected_negative_output)
    evidence(
        "single_step_exact_values",
        {
            "scores": [dot(REFERENCE_CENTER, REFERENCE_POSITIVE), dot(REFERENCE_CENTER, REFERENCE_NEGATIVE)],
            "loss": expected_loss,
            "updated_input": expected_input,
        },
        {
            "scores": detail["scores"],
            "loss": detail["loss"],
            "updated_input": inputs[0].tolist(),
        },
        "independent logistic loss and chain-rule gradients must agree",
    )


def test_rejected_negative_equal_to_positive_is_validation_failure(evidence):
    inputs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    outputs = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    before_input = inputs.copy()
    before_output = outputs.copy()

    try:
        negative_sampling_step(
            inputs,
            outputs,
            center_id=0,
            positive_id=1,
            negative_ids=[1],
            learning_rate=0.1,
        )
    except Exception as exc:
        assert exc.category == "validation_error"
        assert np.array_equal(inputs, before_input)
        assert np.array_equal(outputs, before_output)
        evidence(
            "collision_rejected_without_mutation",
            "validation_error and unchanged matrices",
            {"category": exc.category, "changed": False},
            "a colliding negative has ambiguous label and must be rejected before updates",
        )
    else:
        raise AssertionError("positive/negative collision must not train")


def test_repeated_negative_gradients_accumulate_rather_than_overwrite(evidence):
    inputs = np.array([REFERENCE_CENTER], dtype=np.float64)
    outputs = np.array([REFERENCE_POSITIVE, REFERENCE_NEGATIVE], dtype=np.float64)
    negative_sampling_step(
        inputs,
        outputs,
        center_id=0,
        positive_id=0,
        negative_ids=[1, 1],
        learning_rate=LEARNING_RATE,
    )

    negative = reference_step(
        REFERENCE_CENTER, REFERENCE_NEGATIVE, 0.0, LEARNING_RATE
    )
    expected = [
        value + 2.0 * gradient
        for value, gradient in zip(REFERENCE_NEGATIVE, negative["output_gradient"])
    ]
    assert outputs[1].tolist() == approx(expected)
    evidence(
        "repeated_negative_accumulation",
        expected,
        outputs[1].tolist(),
        "duplicate negative rows represent two samples and must add both gradients",
    )


def test_stable_sigmoid_handles_extreme_scores():
    values = np.array([-1000.0, 0.0, 1000.0], dtype=np.float64)
    result = stable_sigmoid(values)
    assert result[0] == approx(0.0, abs=1e-300)
    assert result[1] == approx(0.5)
    assert result[2] == approx(1.0)
    assert np.all(np.isfinite(result))
