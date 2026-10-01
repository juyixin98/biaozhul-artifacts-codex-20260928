"""Runtime configuration loaded from process environment."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Probability Calibration Lab"
    database_path: Path = Path("data/calibration_runs.sqlite3")
    decomposition_tolerance: float = 1e-10
    log_level: str = "INFO"
    log_file: Path = Path("logs/app.log")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("CALIBRATION_APP_NAME", cls.model_fields["app_name"].default),
            database_path=Path(os.getenv("CALIBRATION_DB_PATH", cls.model_fields["database_path"].default)),
            decomposition_tolerance=float(
                os.getenv(
                    "CALIBRATION_DECOMPOSITION_TOLERANCE",
                    str(cls.model_fields["decomposition_tolerance"].default),
                )
            ),
            log_level=os.getenv("CALIBRATION_LOG_LEVEL", cls.model_fields["log_level"].default).upper(),
            log_file=Path(os.getenv("CALIBRATION_LOG_FILE", cls.model_fields["log_file"].default)),
        )


settings = Settings.from_env()

