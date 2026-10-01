"""Application configuration loaded from local environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    The default SQLite path is intentionally file-based so request identities and
    traces remain inspectable after local restarts.
    """

    app_name: str = "synthetic-equipment-survival"
    service_version: str = "1.0.0"
    schema_version: str = "survival-analysis.v1"
    kernel_version: str = "km-logrank.v1"
    generator_version: str = "synthetic-equipment-weibull.v1"
    db_path: str = os.getenv("SURVIVAL_DB_PATH", "data/survival.sqlite3")
    log_level: str = os.getenv("SURVIVAL_LOG_LEVEL", "INFO")
