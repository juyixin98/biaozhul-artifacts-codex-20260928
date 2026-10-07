from __future__ import annotations

import pytest

from app.artifacts import artifact_from_json, build_artifact
from app.errors import ArtifactError, ExtrapolationRejected
from app.kernel import fit_pav
from app.prediction import predict_scores
from app.validation import validate_records


def fitted_artifact(extrapolation: str = "clip"):
    points, _ = validate_records(
        [
            {"score": 1.0, "target": 0, "weight": 1.0},
            {"score": 2.0, "target": 1, "weight": 1.0},
            {"score": 3.0, "target": 0, "weight": 1.0},
        ]
    )
    pav = fit_pav(points)
    return build_artifact(
        model_id="m1",
        created_at_utc="2026-10-06T00:00:00Z",
        score_name="risk_score",
        target_name="default_next_30d",
        extrapolation=extrapolation,
        pav=pav,
        input_record_count=3,
        zero_weight_count=0,
        training_fingerprint="fixed",
        schema_version="1.0",
    )


def test_artifact_is_piecewise_constant_and_non_decreasing():
    artifact = fitted_artifact()

    assert artifact.scores == (1.0, 2.0)
    assert artifact.probabilities == (pytest.approx((0.0, 0.5)))
    assert artifact.thresholds == (pytest.approx((1.5,)))
    assert artifact.bins[0].lower_threshold is None
    assert artifact.bins[0].upper_threshold == pytest.approx(1.5)
    assert artifact.bins[1].lower_threshold == pytest.approx(1.5)
    assert artifact.bins[1].upper_threshold is None
    assert artifact.score_name == "risk_score"


def test_round_trip_preserves_artifact():
    artifact = fitted_artifact()
    restored = artifact_from_json(artifact.to_json())

    assert restored.model_id == artifact.model_id
    assert restored.probabilities == artifact.probabilities
    assert restored.score_semantics["target_name"] == "default_next_30d"


def test_score_name_is_bound_to_artifact():
    artifact = fitted_artifact()
    with pytest.raises(ArtifactError) as exc_info:
        predict_scores(artifact, [(0, 2.0, None)], score_name="other_score")
    assert exc_info.value.error_code == "artifact_error"


def test_clip_extrapolation_is_marked_uncertain():
    artifact = fitted_artifact("clip")
    predictions, uncertainties = predict_scores(
        artifact,
        [(0, -10.0, "low"), (1, 1.6, None), (2, 10.0, "high")],
        score_name="risk_score",
    )

    assert [item.probability for item in predictions] == [0.0, 0.5, 0.5]
    assert predictions[0].extrapolated is True
    assert predictions[1].extrapolated is False
    assert predictions[2].extrapolated is True
    assert [item["code"] for item in uncertainties] == [
        "extrapolated_probability",
        "extrapolated_probability",
    ]


def test_raise_extrapolation_fails_with_specific_category():
    artifact = fitted_artifact("raise")
    with pytest.raises(ExtrapolationRejected) as exc_info:
        predict_scores(artifact, [(0, 4.0, None)], score_name="risk_score")
    assert exc_info.value.error_code == "extrapolation_rejected"
    assert exc_info.value.details["index"] == 0
