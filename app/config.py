"""Runtime configuration loaded from process environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    log_path: Path
    service_name: str


def get_settings() -> Settings:
    return Settings(
        log_path=Path(os.getenv("NLS_LOG_PATH", "logs/runs.jsonl")),
        service_name=os.getenv("NLS_SERVICE_NAME", "bounded-nls-trust-region"),
    )
