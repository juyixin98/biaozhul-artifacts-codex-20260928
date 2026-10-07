from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "PAV Calibration Backend"
    app_version: str = "1.0.0"
    artifact_schema_version: str = "1.0"
    database_path: str = os.getenv("CALIBRATION_DB", "data/calibration.db")
    log_level: str = os.getenv("CALIBRATION_LOG_LEVEL", "INFO")


def get_settings() -> Settings:
    return Settings()
