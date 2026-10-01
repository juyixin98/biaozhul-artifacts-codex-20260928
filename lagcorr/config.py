"""Fixed analysis settings and process configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_path: str = os.environ.get(
        "LAGCORR_DB_PATH",
        os.path.join(os.getcwd(), "data", "lagcorr.db"),
    )


def get_settings() -> Settings:
    return Settings()
