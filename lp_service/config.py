"""Runtime configuration loaded from local environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    service_name: str
    log_level: str
    log_request_ids: bool


def get_settings() -> Settings:
    return Settings(
        service_name=os.getenv("LP_SERVICE_NAME", "small-lp-simplex"),
        log_level=os.getenv("LP_LOG_LEVEL", "INFO").upper(),
        log_request_ids=os.getenv("LP_LOG_REQUEST_IDS", "true").lower() == "true",
    )
