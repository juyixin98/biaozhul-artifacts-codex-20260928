"""Environment-backed configuration isolated from the rest of the application."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_path: str = "data/calibration.db"
    log_path: str = "logs/app.jsonl"
    default_n_bins: int = 10

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            database_path=os.getenv("CALIBRATION_DB_PATH", "data/calibration.db"),
            log_path=os.getenv("CALIBRATION_LOG_PATH", "logs/app.jsonl"),
            default_n_bins=int(os.getenv("CALIBRATION_DEFAULT_BINS", "10")),
        )
