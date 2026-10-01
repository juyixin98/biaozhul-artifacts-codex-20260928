import pytest

from core.contracts import AnalysisRequest, CalibrationFitRequest, SampleBatch
from core.errors import EmptyCalibrationBin, ProbabilityRangeError
from core.logging_utils import JsonLineLogger
from core.repository import ALGORITHM_VERSION, RunRepository
from core.service import CalibrationService


def service_for(tmp_path):
    repository = RunRepository(str(tmp_path / "runs.db"))
    logger = JsonLineLogger(str(tmp_path / "events.jsonl"))
    return CalibrationService(repository, logger), repository, logger


def test_successful_run_persists_version_bins_and_input_identity(tmp_path):
    service, repository, _ = service_for(tmp_path)
    request = AnalysisRequest(
        y=[0, 1, 0, 1],
        p=[0.2, 0.2, 0.8, 0.8],
        method="fixed",
        edges=[0.0, 0.5, 1.0],
    )

    result = service.analyze(request)
    stored = repository.get_run(result.run_id)

    assert stored["status"] == "success"
    assert stored["algorithm_version"] == ALGORITHM_VERSION
    assert len(stored["input_hash"]) == 64
    assert len(stored["bins"]) == 2
    assert stored["result"]["decomposition"]["raw_brier"] == result.decomposition.raw_brier


def test_invalid_probability_failure_is_persisted_not_reported_success(tmp_path):
    service, repository, logger = service_for(tmp_path)

    with pytest.raises(ProbabilityRangeError):
        service.analyze_raw({"y": [0], "p": [2.0]})

    failure_logs = [event for event in logger.events if event["event"] == "analysis_failed"]
    assert failure_logs and failure_logs[0]["judgment"] == "failure"
    run_id = failure_logs[0]["run_id"]
    stored = repository.get_run(run_id)
    assert stored["status"] == "failed"
    assert stored["error_category"] == "probability_range_error"
    assert stored["result"] is None


def test_fit_empty_evaluation_bin_failure_category_persisted(tmp_path):
    service, repository, logger = service_for(tmp_path)
    request = CalibrationFitRequest(
        train=SampleBatch(y=[0, 1], p=[0.2, 0.8]),
        evaluate=SampleBatch(y=[1], p=[0.5]),
        method="fixed",
        n_bins=10,
    )

    with pytest.raises(EmptyCalibrationBin):
        service.fit_calibration(request)

    failure_logs = [event for event in logger.events if event["event"] == "fit_failed"]
    assert failure_logs[0]["error_category"] == "empty_calibration_bin"
    stored = repository.get_run(failure_logs[0]["run_id"])
    assert stored["status"] == "failed"
    assert stored["error_category"] == "empty_calibration_bin"


def test_successful_fit_stores_train_and_evaluation_analyses(tmp_path):
    service, repository, _ = service_for(tmp_path)
    request = CalibrationFitRequest(
        train=SampleBatch(y=[0, 1, 0, 1], p=[0.2, 0.2, 0.8, 0.8]),
        evaluate=SampleBatch(y=[1, 0], p=[0.2, 0.8]),
        method="quantile",
        n_bins=2,
    )

    result = service.fit_calibration(request)
    stored = repository.get_run(result.run_id)

    assert result.transformed_probabilities == [0.5, 0.5]
    assert {bin_row["stage"] for bin_row in stored["bins"]} == {"train", "evaluation"}
    assert stored["calibrator"] is not None
