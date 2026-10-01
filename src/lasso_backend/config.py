"""Runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    service_name: str = "lasso-path-backend"
    log_level: str = "INFO"
    max_rows: int = 10_000
    max_features: int = 2_000
    max_nnz: int = 2_000_000


def get_settings() -> Settings:
    return Settings(
        log_level=getenv("LASSO_LOG_LEVEL", "INFO").upper(),
        max_rows=int(getenv("LASSO_MAX_ROWS", "10000")),
        max_features=int(getenv("LASSO_MAX_FEATURES", "2000")),
        max_nnz=int(getenv("LASSO_MAX_NNZ", "2000000")),
    )
