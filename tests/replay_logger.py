"""Structured pytest evidence for reproducing a failing or passing problem."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return value


class ReplayLogger:
    def __init__(self, logs_dir: Path) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.run_number = f"pytest-{timestamp}-{uuid.uuid4().hex[:8]}"
        self.path = logs_dir / f"{self.run_number}.jsonl"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self.write(
            "session_start",
            {
                "run_number": self.run_number,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_number": self.run_number,
            **_json_safe(payload),
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def test_start(self, nodeid: str) -> None:
        self._current_nodeid = nodeid
        self.write("test_start", {"nodeid": nodeid})

    def state(self, label: str, state: dict[str, Any], reason: str) -> None:
        self.write(
            "intermediate_state",
            {"nodeid": self._current_nodeid, "label": label, "state": state,
             "reason": reason},
        )

    def test_finish(self, report: Any) -> None:
        self.write(
            "test_finish",
            {
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "duration": report.duration,
                "longrepr": str(report.longrepr) if report.longrepr else None,
            },
        )

    def close(self) -> None:
        self.write("session_end", {"run_number": self.run_number})
        self._file.close()
