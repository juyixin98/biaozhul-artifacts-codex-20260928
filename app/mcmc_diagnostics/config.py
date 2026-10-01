"""Runtime configuration.

The defaults use only local files. Environment variables are intentionally
limited to paths and conservative diagnostic thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    service_name: str = "mcmc-diagnostics-service"
    api_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    kernel_version: str = "1.0.0"
    algorithm_revision: str = "geyer-initial-positive-rank-rhat-2026"
    min_ess_draws: int = 10
    min_rhat_chains: int = 2
    min_split_draws: int = 4
    rhat_warning_threshold: float = 1.01
    low_relative_ess: float = 0.10
    very_low_relative_ess: float = 0.01
    default_acf_factor: float = 10.0
    default_acf_min_lags: int = 20
    database_path: Path = ROOT_DIR / "data" / "diagnostics.db"
    log_path: Path = ROOT_DIR / "logs" / "service.log"
    persist_results: bool = True


def get_settings() -> Settings:
    """Read process-local configuration once per call."""

    root = Path(os.environ.get("MCMC_DIAG_ROOT", ROOT_DIR)).resolve()
    db_env = os.environ.get("MCMC_DIAG_DB")
    log_env = os.environ.get("MCMC_DIAG_LOG")
    return Settings(
        min_ess_draws=int(os.environ.get("MCMC_MIN_ESS_DRAWS", Settings.min_ess_draws)),
        rhat_warning_threshold=float(
            os.environ.get("MCMC_RHAT_THRESHOLD", Settings.rhat_warning_threshold)
        ),
        low_relative_ess=float(os.environ.get("MCMC_LOW_REL_ESS", Settings.low_relative_ess)),
        very_low_relative_ess=float(
            os.environ.get("MCMC_VERY_LOW_REL_ESS", Settings.very_low_relative_ess)
        ),
        database_path=Path(db_env) if db_env else root / "data" / "diagnostics.db",
        log_path=Path(log_env) if log_env else root / "logs" / "service.log",
        persist_results=_as_bool(os.environ.get("MCMC_PERSIST"), True),
    )
