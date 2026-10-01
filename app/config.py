"""Application configuration.

Configuration is intentionally limited to environment variables with fixed
local defaults.  No external service credentials are needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    service_name: str = "covariance-backend"
    api_version: str = "v1"
    implementation_version: str = "1.0.0"
    database_path: Path = Path(os.getenv("COVARIANCE_DB", BASE_DIR / "data" / "covariance.db"))
    log_level: str = os.getenv("COVARIANCE_LOG_LEVEL", "INFO")
    symmetry_tolerance: float = 1e-10
    eigenvalue_tolerance: float = 1e-10


settings = Settings()
