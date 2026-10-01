from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey_imputation.contracts import AnalysisRequest, GenerateDatasetRequest, SurveyRecord


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def minimal_fixture() -> dict:
    with (FIXTURE_DIR / "minimal_survey.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def minimal_records(minimal_fixture) -> list[SurveyRecord]:
    return [SurveyRecord(**row) for row in minimal_fixture["records"]]


@pytest.fixture
def analysis_request(minimal_records) -> AnalysisRequest:
    return AnalysisRequest(
        request_id="test-mean-request",
        records=minimal_records,
        method="mean_imputation",
        seed=1729,
    )


@pytest.fixture
def generation_request() -> GenerateDatasetRequest:
    return GenerateDatasetRequest(n=200, mechanism="mar", missing_probability=0.2, seed=1729)
