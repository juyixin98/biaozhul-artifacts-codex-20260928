"""Calibrator fit/predict behavior: step function, monotonicity, clamp."""
import numpy as np
import pytest

from calibration_service.app.calibrator import (
    EXTRAPOLATION_POLICY,
    IsotonicCalibrator,
)


def fit_simple():
    # labels violate monotonicity at score 0.4 -> pooled with 0.3.
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    weights = np.ones(5)
    return IsotonicCalibrator.fit(scores, labels, weights, score_field="model_score")


def test_fit_produces_nondecreasing_step_mapping():
    cal = fit_simple()
    assert cal.is_monotone()
    # labels at 0.3 (1.0) and 0.4 (0.0) pool to 0.5
    np.testing.assert_allclose(
        cal.predict(np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
        [0.0, 0.0, 0.5, 0.5, 1.0],
        atol=1e-12,
    )


def test_predict_is_piecewise_constant_between_thresholds():
    cal = fit_simple()
    # block k covers scores (edge_{k-1}, edge_k]
    np.testing.assert_allclose(
        cal.predict(np.array([0.15, 0.25, 0.45])), [0.0, 0.5, 1.0], atol=1e-12
    )


def test_out_of_range_scores_clamp_to_boundary():
    cal = fit_simple()
    assert cal.extrapolation == EXTRAPOLATION_POLICY == "clamp_to_boundary"
    assert cal.predict(np.array([-5.0]))[0] == pytest.approx(0.0)
    assert cal.predict(np.array([99.0]))[0] == pytest.approx(1.0)


def test_duplicate_scores_and_zero_weights_affect_fit():
    scores = np.array([0.2, 0.2, 0.8, 0.8, 0.9])
    labels = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    weights = np.array([1.0, 3.0, 1.0, 1.0, 0.0])  # last row dropped
    cal = IsotonicCalibrator.fit(scores, labels, weights, score_field="model_score")
    summary = cal.validation_summary
    assert summary.n_zero_weight_dropped == 1
    assert summary.n_duplicate_scores_merged == 2
    # merged labels: score 0.2 -> 0.25, score 0.8 -> 0.5; both monotone
    np.testing.assert_allclose(cal.probabilities, [0.25, 0.5], atol=1e-12)
    np.testing.assert_allclose(cal.thresholds, [0.2, 0.8], atol=1e-12)


def test_artifact_roundtrip(tmp_path):
    from calibration_service.app.artifact import ArtifactStore

    store = ArtifactStore(str(tmp_path / "a.db"))
    cal = fit_simple()
    meta = store.save("m1", cal, "0.1.0")
    assert meta["version"] == 1
    store.save("m1", cal, "0.1.0")  # second fit bumps version
    loaded, meta2 = store.load("m1")
    assert meta2["version"] == 2
    assert meta2["score_field"] == "model_score"
    np.testing.assert_allclose(
        loaded.predict(np.array([0.35])), cal.predict(np.array([0.35]))
    )
    with pytest.raises(KeyError):
        store.load("missing-model")


def test_unfitted_calibrator_predict_raises():
    cal = IsotonicCalibrator(score_field="x")
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict(np.array([0.1]))
