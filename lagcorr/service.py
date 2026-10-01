"""Application service connecting contracts, statistics, persistence, and logs."""

from __future__ import annotations

from dataclasses import asdict
import uuid

from .kernel import AnalysisOutcome, InvalidSeriesError, SeriesInput, analyze_lags
from .logging_setup import configure_logging
from .repository import AnalysisRepository, DuplicateRequestError
from .schemas import AnalysisRequestPayload


def run_analysis(
    payload: AnalysisRequestPayload,
    repository: AnalysisRepository,
) -> tuple[str, AnalysisOutcome, dict[str, int]]:
    request_id = payload.request_id or f"req_{uuid.uuid4().hex}"

    outcome = analyze_lags(
        a=SeriesInput(times=payload.a.times, values=payload.a.values, name="A"),
        b=SeriesInput(times=payload.b.times, values=payload.b.values, name="B"),
        lags=payload.lags,
        tolerance=payload.tolerance,
    )

    repository.save_analysis(request_id, outcome)
    status_summary: dict[str, int] = {}
    for result in outcome.results:
        status_summary[result.status] = status_summary.get(result.status, 0) + 1

    configure_logging().info(
        "request_id=%s status=%s accepted=%d rejected=%d unknown=%d tolerance=%.12g lags=%d",
        request_id,
        "completed",
        status_summary.get("accepted", 0),
        status_summary.get("rejected", 0),
        status_summary.get("unknown", 0),
        outcome.tolerance,
        len(outcome.lags),
    )
    return request_id, outcome, status_summary


def serialize_outcome(
    request_id: str,
    outcome: AnalysisOutcome,
    status_summary: dict[str, int],
    include_pairs: bool,
) -> dict:
    body = outcome.to_dict()
    body["request_id"] = request_id
    body["status_summary"] = status_summary

    for result in body["results"]:
        result["diagnosis"] = asdict(
            next(
                original.diagnosis
                for original in outcome.results
                if original.lag == result["lag"]
            )
        )
        if not include_pairs:
            result["pairs"] = []

    return body
