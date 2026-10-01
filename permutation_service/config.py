"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("data/runs.sqlite")
    default_monte_carlo_replications: int = 9_999
    exact_max_combinations: int = 100_000
    requested_exact_limit_cap: int = 100_000_000
    max_values: int = 10_000
    max_monte_carlo_replications: int = 10_000_000
    log_level: str = "INFO"

    @staticmethod
    def _positive_int(name: str, default: int, *, minimum: int = 0) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    @classmethod
    def from_env(cls) -> "Settings":
        db_path = Path(os.environ.get("PERM_DB_PATH", "data/runs.sqlite"))
        log_level = os.environ.get("PERM_LOG_LEVEL", "INFO").upper()
        return cls(
            db_path=db_path,
            default_monte_carlo_replications=cls._positive_int(
                "PERM_DEFAULT_MONTE_CARLO_REPLICATIONS", 9_999, minimum=1
            ),
            exact_max_combinations=cls._positive_int(
                "PERM_EXACT_MAX_COMBINATIONS", 100_000, minimum=0
            ),
            requested_exact_limit_cap=cls._positive_int(
                "PERM_REQUESTED_EXACT_LIMIT_CAP", 100_000_000, minimum=0
            ),
            max_values=cls._positive_int("PERM_MAX_VALUES", 10_000, minimum=2),
            max_monte_carlo_replications=cls._positive_int(
                "PERM_MAX_MONTE_CARLO_REPLICATIONS", 10_000_000, minimum=1
            ),
            log_level=log_level,
        )

