"""Fixed analysis settings and runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


SERVICE_NAME = "irregular-lag-correlation"
SCHEMA_VERSION = "v1"
ALGORITHM_VERSION = "irregular-lag-correlation-core-1.0.0"

# These statistical settings are intentionally fixed rather than request options.
DETREND_METHOD = "linear"
STANDARDIZATION = "sample_z_score"
MIN_PAIRS = 6
CORRELATION_PRECISION = 12
MAX_LAG_POINTS = 10_001


@dataclass(frozen=True)
class Settings:
    database_path: str


def get_settings() -> Settings:
    return Settings(
        database_path=os.environ.get(
            "LAG_CORRELATION_DB_PATH",
            os.path.join(os.getcwd(), "data", "lag_correlation.sqlite3"),
        )
    )
