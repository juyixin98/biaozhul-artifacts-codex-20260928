"""Service configuration, independent from modules and tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    db_path: str = str(BASE_DIR / "data" / "ddservice.db")
    target_program: str = str(BASE_DIR / "fixtures" / "target_program.py")
    run_timeout_sec: float = 5.0
    # Extra environment fingerprint components, so cache entries are bound
    # to the environment they were produced in.
    env_fingerprint_extra: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.environ.get("DDSVC_DB", str(BASE_DIR / "data" / "ddservice.db")),
            target_program=os.environ.get(
                "DDSVC_TARGET", str(BASE_DIR / "fixtures" / "target_program.py")
            ),
            run_timeout_sec=float(os.environ.get("DDSVC_TIMEOUT", "5.0")),
        )


def env_fingerprint(settings: Settings) -> str:
    import platform
    import sys

    parts = [
        sys.version.split()[0],
        platform.system(),
        platform.machine(),
        *settings.env_fingerprint_extra,
    ]
    return "|".join(parts)
