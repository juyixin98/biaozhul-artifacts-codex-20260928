"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    work_root: Path
    default_baseline_timeout: float
    default_mutant_timeout: float
    log_tail_characters: int


def get_settings() -> Settings:
    db_raw = os.getenv("MUTATION_DB_PATH", "data/mutation-evidence.sqlite3")
    return Settings(
        database_path=Path(db_raw),
        work_root=Path(
            os.getenv(
                "MUTATION_WORK_ROOT",
                str(Path(tempfile.gettempdir()) / "python-mutation-backend-runs"),
            )
        ),
        default_baseline_timeout=float(os.getenv("MUTATION_BASELINE_TIMEOUT", "10")),
        default_mutant_timeout=float(os.getenv("MUTANT_TIMEOUT", "1")),
        log_tail_characters=int(os.getenv("MUTATION_LOG_TAIL_CHARS", "4000")),
    )
