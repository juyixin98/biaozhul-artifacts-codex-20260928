"""Central configuration for the mutation testing backend."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Values may be overridden via environment variables."""

    db_path: str = os.environ.get("MUTATION_DB_PATH", "mutation_evidence.db")
    work_root: str = os.environ.get("MUTATION_WORK_ROOT", "")  # empty -> system temp
    default_timeout_seconds: float = float(
        os.environ.get("MUTATION_DEFAULT_TIMEOUT", "30")
    )
    baseline_timeout_seconds: float = float(
        os.environ.get("MUTATION_BASELINE_TIMEOUT", "60")
    )
    max_mutants_per_run: int = int(os.environ.get("MUTATION_MAX_MUTANTS", "500"))
    operator_set_version: str = "ops-v1"  # bump when operator semantics change


def get_settings() -> Settings:
    return Settings()
