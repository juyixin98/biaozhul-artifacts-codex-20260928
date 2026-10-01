"""Configuration loaded from process environment.

The service deliberately has no external configuration service. All paths and
limits are local development settings with explicit, testable parsers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("data/changepoint.db")
    exact_dp_max_samples: int = 2000
    pruned_dp_max_samples: int = 120
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        log_level = os.environ.get("CHANGPOINT_LOG_LEVEL", "INFO").upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if log_level not in allowed:
            raise ValueError(
                f"CHANGPOINT_LOG_LEVEL must be one of {sorted(allowed)}"
            )
        return cls(
            db_path=Path(
                os.environ.get(
                    "CHANGPOINT_DB_PATH", "data/changepoint.db"
                )
            ),
            exact_dp_max_samples=_env_int(
                "CHANGPOINT_EXACT_DP_MAX_SAMPLES", 2000
            ),
            pruned_dp_max_samples=_env_int(
                "CHANGPOINT_PRUNED_DP_MAX_SAMPLES", 120
            ),
            log_level=log_level,
        )
