from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_path: str = "data/change_detection.db"
    max_series_length: int = 1000
    log_level: str = "INFO"
    log_path: str = "logs/app.log"

    @classmethod
    def from_env(cls) -> "Settings":
        max_length = int(os.getenv("CHG_MAX_SERIES_LENGTH", "1000"))
        if max_length < 1:
            raise ValueError("CHG_MAX_SERIES_LENGTH must be at least 1")
        return cls(
            database_path=os.getenv("CHG_DATABASE_PATH", "data/change_detection.db"),
            max_series_length=max_length,
            log_level=os.getenv("CHG_LOG_LEVEL", "INFO").upper(),
            log_path=os.getenv("CHG_LOG_PATH", "logs/app.log"),
        )

