import pytest

from core.calibration import HistogramCalibrator
from core.contracts import SampleBatch
from core.errors import EmptyCalibrationBin
from core.metrics import NumpySamples


def test_calibrator_fits_only_on_training_probabilities():
    train = NumpySamples.from_contract(SampleBatch(y=[0, 1, 0, 1], p=[0.2, 0.2, 0.8, 0.8]))
    calibrator = HistogramCalibrator.fit(train.p, train.y, train.w, "quantile", 2)

    assert calibrator.edges.tolist() == [0.0, 0.5, 1.0]
    assert calibrator.bin_predicted_rates == (0.5, 0.5)

    transformed = calibrator.transform([0.19, 0.21, 0.79, 0.81])
    assert transformed.tolist() == [0.5, 0.5, 0.5, 0.5]


def test_empty_fitted_bin_rejects_evaluation_probability_without_smoothing():
    train = NumpySamples.from_contract(SampleBatch(y=[0, 1], p=[0.2, 0.8]))
    calibrator = HistogramCalibrator.fit(
        train.p,
        train.y,
        train.w,
        "fixed",
        n_bins=10,
    )

    with pytest.raises(EmptyCalibrationBin) as excinfo:
        calibrator.transform([0.5])

    assert excinfo.value.category == "empty_calibration_bin"
    assert "empty fitted bin" in str(excinfo.value)


def test_evaluation_is_separate_from_train_contract():
    train = SampleBatch(y=[0, 1, 1, 0], p=[0.1, 0.2, 0.8, 0.9])
    evaluation = SampleBatch(y=[1, 0], p=[0.2, 0.8])

    train_samples = NumpySamples.from_contract(train)
    calibrator = HistogramCalibrator.fit(train_samples.p, train_samples.y, train_samples.w, "fixed", 2)
    evaluation_samples = NumpySamples.from_contract(evaluation)

    transformed = calibrator.transform(evaluation_samples.p)
    assert transformed.tolist() == [0.5, 0.5]
    assert evaluation.p == [0.2, 0.8]


def test_probability_boundaries_are_still_rejected_in_transform_contract():
    from core.errors import ProbabilityRangeError

    with pytest.raises(ProbabilityRangeError):
        SampleBatch(y=[0], p=[1.1])
