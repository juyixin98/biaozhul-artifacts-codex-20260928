"""Configuration layer.

Settings live in a frozen dataclass with environment overrides, so tests can
inject isolated instances and deployments can be tuned without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

_ENV_PREFIX = "PERMTEST_"


@dataclass(frozen=True)
class Settings:
    # Combination-space size at or below which the kernel enumerates every
    # permutation exactly instead of sampling.
    exact_threshold: int = 50_000
    # Monte Carlo defaults and bounds.
    default_n_permutations: int = 10_000
    min_n_permutations: int = 100
    max_n_permutations: int = 2_000_000
    # Guard rail on combined input size.
    max_group_total: int = 10_000
    # Null distributions larger than this are never shipped in responses.
    distribution_max_size: int = 10_000
    # Permutations evaluated per kernel chunk (progress/cancel granularity).
    chunk_size: int = 2_048
    # SQLite location for the versioned run store.
    db_path: str = "permtest.db"


def load_settings() -> Settings:
    """Build settings from environment variables named PERMTEST_<FIELD>."""
    int_fields = {
        "exact_threshold",
        "default_n_permutations",
        "min_n_permutations",
        "max_n_permutations",
        "max_group_total",
        "distribution_max_size",
        "chunk_size",
    }
    overrides: dict = {}
    for f in fields(Settings):
        raw = os.environ.get(f"{_ENV_PREFIX}{f.name.upper()}")
        if raw is None:
            continue
        if f.name in int_fields:
            try:
                overrides[f.name] = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"environment variable {_ENV_PREFIX}{f.name.upper()} must be an integer, got {raw!r}"
                ) from exc
        else:
            overrides[f.name] = raw
    return Settings(**overrides)
