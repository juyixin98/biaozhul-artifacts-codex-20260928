"""Runtime configuration.

The service intentionally has no cloud or database credentials.  Dataset state is
kept in an in-memory store and all fixtures are synthetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "Incremental Nondominated Sorting API"
    api_prefix: str = "/api/v1"
    api_version: str = "v1"
    algorithm_version: str = "exact-domination-graph-v1"
    default_tolerance: float = 0.0


def load_settings() -> Settings:
    tolerance = float(os.getenv("NDS_TOLERANCE", "0.0"))
    if tolerance < 0.0:
        raise ValueError("NDS_TOLERANCE must be non-negative")
    return Settings(default_tolerance=tolerance)


settings = load_settings()

