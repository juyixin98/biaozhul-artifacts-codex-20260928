"""Structured JSONL run journals used for replay and postmortem debugging."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import StateConflictError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_run_id() -> str:
    return f"milp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:10]}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        if value != value:
            return "NaN"
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class RunJournal:
    def __init__(self, run_id: str, journal_dir: Path, problem: dict[str, Any], options: dict[str, Any]) -> None:
        self.run_id = run_id
        self.journal_dir = journal_dir
        self.path = journal_dir / f"{run_id}.jsonl"
        if self.path.exists():
            raise StateConflictError(f"run_id already exists: {run_id}")
        self.sequence = 0
        journal_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "record": "run_start",
                        "timestamp": utc_now_iso(),
                        "run_id": run_id,
                        "problem": problem,
                        "options": options,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n"
            )

    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        event = {"record": "event", "timestamp": utc_now_iso(), "sequence": self.sequence, **payload}
        event = json_safe(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        return event

    def complete(self, result: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    json_safe(
                        {"record": "run_complete", "timestamp": utc_now_iso(), "run_id": self.run_id, "result": result}
                    ),
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
