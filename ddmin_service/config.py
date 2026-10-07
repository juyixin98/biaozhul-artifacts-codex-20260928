"""Service configuration, overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str = "ddmin_service.db"
    default_timeout_seconds: float = 5.0
    # 0 means unlimited; a positive value pauses a run after that many
    # live (non-cached) program executions, enabling interrupt/resume.
    default_live_query_budget: int = 0

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            db_path=os.environ.get("DD_DB_PATH", "ddmin_service.db"),
            default_timeout_seconds=float(os.environ.get("DD_TIMEOUT_SECONDS", "5")),
            default_live_query_budget=int(os.environ.get("DD_LIVE_QUERY_BUDGET", "0")),
        )
