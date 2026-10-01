"""JSONL run records used to reproduce and diagnose solver decisions."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


class RunRecorder:
    """Append one JSON object per line for one solver run."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        run_id: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.run_id = run_id or new_run_id()
        self.log_path = self.log_dir / f"{self.run_id}.jsonl"
        self.started = False
        self.finished = False
        self._lock = threading.Lock()

    def start(self, payload: dict[str, Any]) -> None:
        if self.started:
            return
        self.started = True
        self.log_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "start",
            "run_id": self.run_id,
            "time": utc_now_iso(),
            "pid": os.getpid(),
        }
        event.update(payload)
        self._write(event)

    def iteration(
        self,
        iteration: int,
        state: dict[str, Any],
        *,
        total_iterations: int,
    ) -> None:
        # Keep enough states to replay a small or slowly converging run without
        # turning large successful runs into oversized logs.
        is_head = iteration <= 5
        is_tail = iteration > total_iterations - 5
        is_power_of_two = iteration > 1 and (iteration & (iteration - 1)) == 0
        if not (is_head or is_tail or is_power_of_two or total_iterations <= 50):
            return
        self.event(
            "iteration",
            iteration=iteration,
            total_iterations=total_iterations,
            state=state,
        )

    def event(self, event: str, **payload: Any) -> None:
        self._write(
            {
                "event": event,
                "run_id": self.run_id,
                "time": utc_now_iso(),
                **payload,
            }
        )

    def finish(self, result: dict[str, Any]) -> None:
        if self.finished:
            return
        self.finished = True
        self.event("finish", result=result)

    def failure(self, category: str, message: str, details: dict[str, Any]) -> None:
        if self.finished:
            return
        self.finished = True
        self.event(
            "failure",
            category=category,
            message=message,
            details=details,
        )

    def _write(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":"))
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def read_events(self) -> list[dict[str, Any]]:
        with self.log_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

