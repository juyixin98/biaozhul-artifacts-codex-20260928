from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path(
        os.getenv("COVARIANCE_DB", str(BASE_DIR / "data" / "covariance.sqlite3"))
    )
    max_dimension: int = int(os.getenv("COVARIANCE_MAX_DIM", "120"))
    max_samples: int = int(os.getenv("COVARIANCE_MAX_SAMPLES", "2000"))


settings = Settings()
