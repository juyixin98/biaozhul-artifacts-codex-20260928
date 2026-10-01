"""Application configuration loaded from the process environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError


def _as_float(name: str, raw: str, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite number") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ConfigurationError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum:g}")
    return value


@dataclass(frozen=True)
class Settings:
    journal_dir: Path = Path("logs")
    default_integer_tolerance: float = 1e-8
    default_constraint_tolerance: float = 1e-8
    default_lp_tolerance: float = 1e-10
    default_time_limit_seconds: float = 30.0
    default_node_limit: int = 10_000
    max_events_in_response: int = 200

    @classmethod
    def from_env(cls) -> "Settings":
        journal_dir = Path(os.getenv("MILP_JOURNAL_DIR", "logs"))

        int_tol = _as_float(
            "MILP_INTEGER_TOLERANCE",
            os.getenv("MILP_INTEGER_TOLERANCE", "1e-8"),
            minimum=0.0,
        )
        constraint_tol = _as_float(
            "MILP_CONSTRAINT_TOLERANCE",
            os.getenv("MILP_CONSTRAINT_TOLERANCE", "1e-8"),
            minimum=0.0,
        )
        lp_tol = _as_float(
            "MILP_LP_TOLERANCE",
            os.getenv("MILP_LP_TOLERANCE", "1e-10"),
            minimum=0.0,
        )
        default_time = _as_float(
            "MILP_DEFAULT_TIME_LIMIT_SECONDS",
            os.getenv("MILP_DEFAULT_TIME_LIMIT_SECONDS", "30"),
            minimum=0.0,
        )
        max_events_raw = os.getenv("MILP_MAX_EVENTS_IN_RESPONSE", "200")
        try:
            max_events = int(max_events_raw)
        except ValueError as exc:
            raise ConfigurationError("MILP_MAX_EVENTS_IN_RESPONSE must be an integer") from exc
        if max_events < 1:
            raise ConfigurationError("MILP_MAX_EVENTS_IN_RESPONSE must be >= 1")

        node_limit_raw = os.getenv("MILP_DEFAULT_NODE_LIMIT", "10000")
        try:
            node_limit = int(node_limit_raw)
        except ValueError as exc:
            raise ConfigurationError("MILP_DEFAULT_NODE_LIMIT must be an integer") from exc
        if node_limit < 1:
            raise ConfigurationError("MILP_DEFAULT_NODE_LIMIT must be >= 1")

        return cls(
            journal_dir=journal_dir,
            default_integer_tolerance=int_tol,
            default_constraint_tolerance=constraint_tol,
            default_lp_tolerance=lp_tol,
            default_time_limit_seconds=default_time,
            default_node_limit=node_limit,
            max_events_in_response=max_events,
        )


_cached_settings: Settings | None = None


def get_settings() -> Settings:
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings.from_env()
    return _cached_settings
