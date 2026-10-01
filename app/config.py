"""Application configuration loaded from process environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = "Synthetic Survival Analysis API"
    environment: str = "local"
    sqlite_path: Path = BASE_DIR / "data" / "survival_requests.db"
    log_level: str = "INFO"


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("SURVIVAL_ENV", "local"),
        sqlite_path=_path_from_env("SURVIVAL_SQLITE_PATH", Settings.sqlite_path),
        log_level=os.getenv("SURVIVAL_LOG_LEVEL", Settings.log_level).upper(),
    )
