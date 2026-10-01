"""Runtime settings.

The default SQLite database is a local project file. No external service is
required to reproduce the case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_path: Path
    results_dir: Path
    redact_sensitive: bool = True


def get_settings() -> Settings:
    db_env = os.getenv("SURVEY_DB_PATH")
    results_env = os.getenv("SURVEY_RESULTS_DIR")
    return Settings(
        database_path=Path(db_env) if db_env else PROJECT_ROOT / "data" / "survey.db",
        results_dir=Path(results_env) if results_env else PROJECT_ROOT / "results",
        redact_sensitive=os.getenv("SURVEY_REDACT", "1") != "0",
    )
