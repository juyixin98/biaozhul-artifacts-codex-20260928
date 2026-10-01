"""Process-local server configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class ServerSettings:
    data_dir: Path
    max_runs: int = 256
    max_problem_dim: int = 256
    max_iterations: int = 1000
    max_event_log_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, Path):
            object.__setattr__(self, "data_dir", Path(self.data_dir))

    @staticmethod
    def from_env() -> "ServerSettings":
        return ServerSettings(
            data_dir=Path(os.getenv("ADMM_DATA_DIR", ".local_data/admm")),
            max_runs=_int_env("ADMM_MAX_RUNS", 256),
            max_problem_dim=_int_env("ADMM_MAX_PROBLEM_DIM", 256),
            max_iterations=_int_env("ADMM_MAX_ITERATIONS", 1000),
            max_event_log_bytes=_int_env(
                "ADMM_MAX_EVENT_LOG_BYTES", 64 * 1024 * 1024
            ),
        )
