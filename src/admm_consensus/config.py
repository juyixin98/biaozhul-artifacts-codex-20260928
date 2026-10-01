"""Runtime limits with explicit resource-exhaustion boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_dimension: int
    max_runs: int
    max_iterations: int
    max_workers: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("ADMM_DATA_DIR", "./.admm_runs")).resolve()
        return cls(
            data_dir=data_dir,
            max_dimension=int(os.getenv("ADMM_MAX_DIMENSION", "64")),
            max_runs=int(os.getenv("ADMM_MAX_RUNS", "128")),
            max_iterations=int(os.getenv("ADMM_MAX_ITERATIONS", "1000")),
            max_workers=int(os.getenv("ADMM_MAX_WORKERS", "2")),
        )
