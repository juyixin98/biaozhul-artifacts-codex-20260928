from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    max_predictive_trials: int = 10_000
    max_observation_count: int = 10**15
    max_prior_parameter: float = 1.0e100
    interval_level: float = 0.95


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    default_db = Path(__file__).resolve().parent.parent / "data" / "beta_binomial.sqlite3"
    db_path = Path(os.environ.get("BETA_BINOMIAL_DB_PATH", str(default_db)))
    return Settings(
        database_path=db_path,
        max_predictive_trials=_env_int("BETA_BINOMIAL_MAX_PREDICTIVE_TRIALS", 10_000),
        max_observation_count=_env_int("BETA_BINOMIAL_MAX_OBSERVATION_COUNT", 10**15),
        max_prior_parameter=_env_float("BETA_BINOMIAL_MAX_PRIOR_PARAMETER", 1.0e100),
    )

