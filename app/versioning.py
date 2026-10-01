"""Stable response envelopes and version metadata."""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.errors import FailureRecord


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def processing_location() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "pid": os.getpid(),
        "service": settings.service_name,
        "module_root": os.path.dirname(os.path.abspath(__file__)),
    }


def envelope(
    *,
    request_id: str,
    status: str,
    sample_report: dict[str, Any] | None,
    results: list[dict[str, Any]],
    failures: list[FailureRecord],
    uncertainties: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": status,
        "created_at": utc_now(),
        "version": {
            "api": settings.api_version,
            "implementation": settings.implementation_version,
            "algorithm_manifest": {
                "empirical": "MLE covariance, divisor n, arithmetic mean",
                "ledoit_wolf": "Ledoit-Wolf 2004 scalar identity target, divisor n",
                "ogk": "Maronna-Zamar two-step raw OGK with MAD and GK scatter",
            },
        },
        "processing_location": processing_location(),
        "sample": sample_report,
        "results": results,
        "failures": [failure.to_dict() for failure in failures],
        "uncertainties": uncertainties,
    }


def error_envelope(
    *,
    request_id: str,
    failure: FailureRecord,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": "failed",
        "created_at": utc_now(),
        "version": {
            "api": settings.api_version,
            "implementation": settings.implementation_version,
        },
        "processing_location": processing_location(),
        "request": payload,
        "failures": [failure.to_dict()],
        "uncertainties": [
            {
                "scope": "request",
                "conclusion": "No covariance result was produced because the request failed before estimation.",
            }
        ],
    }
