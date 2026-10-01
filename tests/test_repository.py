from __future__ import annotations

import pytest

from experiments.reference import scenarios
from lag_correlation.config import ALGORITHM_VERSION, SCHEMA_VERSION
from lag_correlation.errors import DuplicateRequestError, ResultNotFoundError
from lag_correlation.kernel.analysis import analyze_lag_correlation
from lag_correlation.kernel.models import Series
from lag_correlation.repository import AnalysisRepository


def test_repository_persists_result_and_sanitized_diagnostics(tmp_path) -> None:
    scenario = next(
        item for item in scenarios() if item.name == "known_offset_different_density"
    )
    result = analyze_lag_correlation(
        x=Series(scenario.x_name, scenario.x_times, scenario.x_values),
        y=Series(scenario.y_name, scenario.y_times, scenario.y_values),
        min_lag=scenario.lag_grid[0],
        max_lag=scenario.lag_grid[1],
        step=scenario.lag_grid[2],
        tolerance=scenario.tolerance,
        request_id="repo-success",
    )
    repository = AnalysisRepository(str(tmp_path / "analysis.sqlite3"))

    repository.save(result)
    stored = repository.get_response_dict("repo-success")

    assert stored["request_id"] == "repo-success"
    assert stored["status"] == "ok"
    assert stored["peak_lag"] == 2.0
    assert stored["schema_version"] == SCHEMA_VERSION
    assert stored["algorithm_version"] == ALGORITHM_VERSION
    assert stored["best_lag"]["pair_details"] is None
    assert stored["diagnostics"][0]["request_id"] == "repo-success"
    assert all("values" not in event for event in stored["diagnostics"])
    assert "does not establish causality" in stored["causality_notice"]


def test_repository_rejects_duplicate_request_id(tmp_path) -> None:
    scenario = next(item for item in scenarios() if item.name == "constant_after_detrend")
    result = analyze_lag_correlation(
        x=Series(scenario.x_name, scenario.x_times, scenario.x_values),
        y=Series(scenario.y_name, scenario.y_times, scenario.y_values),
        min_lag=0.0,
        max_lag=0.0,
        step=1.0,
        tolerance=scenario.tolerance,
        request_id="duplicate-id",
    )
    repository = AnalysisRepository(str(tmp_path / "analysis.sqlite3"))
    repository.save(result)

    with pytest.raises(DuplicateRequestError, match="duplicate-id"):
        repository.save(result)


def test_repository_missing_result_has_specific_failure_category(tmp_path) -> None:
    repository = AnalysisRepository(str(tmp_path / "analysis.sqlite3"))

    with pytest.raises(ResultNotFoundError) as error:
        repository.get_response_dict("missing")

    assert error.value.code == "RESULT_NOT_FOUND"
