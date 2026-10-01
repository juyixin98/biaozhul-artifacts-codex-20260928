"""Standalone solver configuration.

Defaults live here; every field can be overridden through environment
variables prefixed with ``BILP_`` (e.g. ``BILP_MAX_NODES=500``) or, for a
subset of budget knobs, per request via solve options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Mapping


@dataclass(frozen=True)
class SolverConfig:
    # Numerical tolerances.
    int_tol: float = 1e-6            # |x - round(x)| <= int_tol  => integer candidate
    candidate_feas_tol: float = 1e-7  # feasibility tolerance for the *rounded* candidate
    prune_tol: float = 1e-9          # bound-vs-incumbent comparison tolerance
    # Default solve budgets.
    max_nodes: int = 10_000
    time_limit_s: float = 30.0
    # Admission limits (problem size).
    max_variables: int = 200
    max_constraints: int = 500
    # Hard caps for per-request budget overrides.
    hard_max_nodes: int = 200_000
    hard_max_time_s: float = 300.0
    # Evidence retention.
    max_evidence_nodes: int = 1_000
    # Optional JSONL persistence of runs (for replay across restarts).
    runs_log_path: str | None = None
    # LP solver backing the relaxations.
    lp_method: str = "highs"


def load_config(env: Mapping[str, str] | None = None) -> SolverConfig:
    """Build a config from environment variables (prefix ``BILP_``)."""
    env = os.environ if env is None else env
    kwargs: dict[str, object] = {}
    for f in fields(SolverConfig):
        key = f"BILP_{f.name.upper()}"
        raw = env.get(key)
        if raw is None:
            continue
        if f.type == "int" or f.name in ("max_nodes", "max_variables", "max_constraints",
                                         "hard_max_nodes", "max_evidence_nodes"):
            kwargs[f.name] = int(raw)
        elif f.name in ("int_tol", "candidate_feas_tol", "prune_tol",
                        "time_limit_s", "hard_max_time_s"):
            kwargs[f.name] = float(raw)
        else:
            kwargs[f.name] = raw
    return SolverConfig(**kwargs)
