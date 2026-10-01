"""JSON-lines run logging with stable run and error categories."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JsonLineLogger:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []

    def log(self, event: str, run_id: str | None = None, **fields: Any) -> dict[str, Any]:
        record = {
            "timestamp": utc_now_iso(),
            "event": event,
            "run_id": run_id,
            **fields,
        }
        self.events.append(record)
        if self.path:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        return record
