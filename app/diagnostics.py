"""Structured run diagnostics and JSONL logging."""

from __future__ import annotations

import json
import logging
import os
import platform
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


@dataclass
class IterationRecord:
    iteration: int
    cost: float
    scaled_projected_gradient_norm: float
    trust_radius: float
    step_norm: float
    predicted_reduction: float
    actual_reduction: float
    reduction_ratio: float
    accepted: bool
    decision: str
    active_lower: list[str]
    active_upper: list[str]
    interior_stationary: bool
    boundary_stationary: bool


@dataclass
class SolveDiagnostics:
    run_id: str
    expression: str
    parameter_names: list[str]
    parameter_scales: list[float]
    scale_source: dict[str, str]
    observations: int
    max_iterations: int
    tolerance: float
    iterations: list[IterationRecord] = field(default_factory=list)
    status: str = "MAX_ITERATIONS"
    reason: str = ""
    rank: int | None = None
    singular_values: list[float] | None = None
    condition_number: float | None = None
    nullspace: list[list[float]] | None = None
    finite_difference_max_error: float | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def append(self, record: IterationRecord) -> None:
        self.iterations.append(record)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["iterations"] = [asdict(item) for item in self.iterations]
        return json_safe(data)


def default_log_path() -> Path:
    return Path(os.getenv("NLS_LOG_PATH", "logs/runs.jsonl"))


def package_versions() -> dict[str, str]:
    import fastapi
    import numpy
    import scipy

    return {
        "service": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "fastapi": fastapi.__version__,
    }


class RunLogger:
    """JSONL logger where every line carries the same run identity."""

    def __init__(self, path: str | Path | None = None, run_id: str | None = None):
        self.path = Path(path) if path is not None else default_log_path()
        self.run_id = run_id or str(uuid.uuid4())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"bounded_nls.{self.run_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not any(getattr(handler, "_nls_jsonl", False) for handler in self._logger.handlers):
            handler = logging.FileHandler(self.path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler._nls_jsonl = True  # type: ignore[attr-defined]
            self._logger.addHandler(handler)

    def event(self, event: str, **payload: Any) -> None:
        message = {
            "run_id": self.run_id,
            "event": event,
            "versions": package_versions(),
            **json_safe(payload),
        }
        self._logger.info(json.dumps(message, ensure_ascii=False, sort_keys=True))
