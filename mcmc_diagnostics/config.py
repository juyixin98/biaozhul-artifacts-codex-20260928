"""Runtime configuration loaded from process-local environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    max_chains: int = 100
    max_draws_per_chain: int = 100_000
    default_max_acf_lag: int = 50
    rhat_warning_threshold: float = 1.01
    ess_low_fraction: float = 0.10


def get_settings() -> Settings:
    max_acf_lag = int(os.getenv("MCMC_MAX_ACF_LAG", "50"))
    if max_acf_lag < 0:
        raise ValueError("MCMC_MAX_ACF_LAG must be non-negative")
    return Settings(
        database_path=Path(os.getenv("MCMC_DB_PATH", "./data/diagnostics.db")),
        max_chains=int(os.getenv("MCMC_MAX_CHAINS", "100")),
        max_draws_per_chain=int(os.getenv("MCMC_MAX_DRAWS", "100000")),
        default_max_acf_lag=max_acf_lag,
        rhat_warning_threshold=float(
            os.getenv("MCMC_RHAT_WARNING_THRESHOLD", "1.01")
        ),
        ess_low_fraction=float(os.getenv("MCMC_ESS_LOW_FRACTION", "0.10")),
    )
