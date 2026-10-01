"""Immutable decomposition and service limits."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StlConfig:
    period: int = 7
    seasonal_window: int = 7
    trend_window: int = 15
    low_pass_window: int = 9
    seasonal_degree: int = 1
    trend_degree: int = 1
    low_pass_degree: int = 1
    inner_iterations: int = 2
    outer_iterations: int = 15
    max_observations: int = 100_000
    kernel_version: str = "constrained-stl-numpy-1.0.0"
    schema_version: str = "decomposition-response-v1"

    @property
    def minimum_observations(self) -> int:
        # Two full periods are required for seasonal centering and a trend fit.
        return self.period * 2

    def as_public_dict(self) -> dict:
        return {
            "period": self.period,
            "seasonal_window": self.seasonal_window,
            "trend_window": self.trend_window,
            "low_pass_window": self.low_pass_window,
            "seasonal_degree": self.seasonal_degree,
            "trend_degree": self.trend_degree,
            "low_pass_degree": self.low_pass_degree,
            "inner_iterations": self.inner_iterations,
            "outer_iterations": self.outer_iterations,
            "minimum_observations": self.minimum_observations,
            "maximum_observations": self.max_observations,
        }


DEFAULT_CONFIG = StlConfig()
DEFAULT_DATABASE_PATH = os.environ.get(
    "STL_BACKEND_DB", str(os.path.join(os.getcwd(), "stl_runs.sqlite3"))
)
