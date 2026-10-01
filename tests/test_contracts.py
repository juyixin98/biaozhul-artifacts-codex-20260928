from __future__ import annotations

import pytest
from pydantic import ValidationError

from lag_correlation.contracts import AnalysisRequest, LagGrid, MatchingWindow, SeriesPayload


def make_request(**overrides) -> AnalysisRequest:
    data = {
        "request_id": "contract-test",
        "x": SeriesPayload(name="x", times=[0.0, 1.0], values=[0.0, 1.0]),
        "y": SeriesPayload(name="y", times=[0.0, 1.0], values=[1.0, 0.0]),
        "lag_grid": LagGrid(min_lag=-1.0, max_lag=1.0, step=0.5),
        "matching": MatchingWindow(tolerance=0.1),
    }
    data.update(overrides)
    return AnalysisRequest(**data)


def test_series_rejects_non_monotonic_times() -> None:
    with pytest.raises(ValidationError) as error:
        SeriesPayload(name="bad", times=[0.0, 0.0], values=[1.0, 2.0])

    assert "strictly increasing" in str(error.value)


def test_series_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError) as ValidationErrorInfo:
        SeriesPayload(name="bad", times=[0.0, 1.0], values=[1.0])

    assert "equal length" in str(ValidationErrorInfo.value)


def test_lag_grid_rejects_too_many_points() -> None:
    with pytest.raises(ValidationError) as error:
        LagGrid(min_lag=0.0, max_lag=20_000.0, step=1.0)

    assert "at most" in str(error.value)


def test_valid_request_accepts_explicit_contract() -> None:
    request = make_request()

    assert request.request_id == "contract-test"
    assert request.matching.tolerance == 0.1
