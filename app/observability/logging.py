"""Structured run-scoped logging."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from typing import Any

import numpy
import scipy

from app import __version__


def _json_default(value: Any) -> Any:
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (numpy.floating, numpy.integer)):
        return value.item()
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return str(value)


class RunLogger:
    """Attach run identity, versions, and numerical progress to every event."""

    def __init__(self, run_id: str, logger: logging.Logger | None = None) -> None:
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.logger = logger or logging.getLogger("app.nls")

    def event(self, stage: str, **payload: Any) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": stage,
            **payload,
        }
        self.events.append(event)
        self.logger.info(json.dumps(event, default=_json_default, sort_keys=True))
        return event

    def log_start(self, expression: str, n_points: int, parameter_names: list[str]) -> None:
        self.event(
            "run_start",
            service_version=__version__,
            python_version=platform.python_version(),
            numpy_version=numpy.__version__,
            scipy_version=scipy.__version__,
            expression=expression,
            n_points=n_points,
            parameter_names=parameter_names,
        )
