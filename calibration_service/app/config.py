"""Service configuration. Values come from environment variables with
local-development defaults; no production credentials are required."""
from __future__ import annotations

import os
from dataclasses import dataclass

LIB_VERSION = "0.1.0"


@dataclass(frozen=True)
class Settings:
    db_path: str = os.environ.get("CALIBRATION_DB", "./calibration_artifacts.db")
    score_field: str = os.environ.get("CALIBRATION_SCORE_FIELD", "model_score")
    ece_bins: int = int(os.environ.get("CALIBRATION_ECE_BINS", "10"))


settings = Settings()
