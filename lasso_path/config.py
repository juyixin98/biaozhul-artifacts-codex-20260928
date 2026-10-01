"""Runtime configuration.

Defaults are frozen algorithmic choices; every value can be overridden through
environment variables with the ``LASSO_`` prefix (e.g. ``LASSO_DEFAULT_TOL``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LASSO_", extra="ignore")

    # Solver tolerances. Convergence is declared when
    #   duality_gap <= default_tol * max(1, ||y_c||^2 / (2 n)).
    default_tol: float = 1e-4
    default_max_iter: int = 1000

    # Automatically generated lambda path.
    default_n_lambdas: int = 100
    max_n_lambdas: int = 500
    lambda_min_ratio: float = 1e-3

    # Columns with population variance <= zero_variance_eps are treated as
    # constant (rejected or dropped depending on the request policy).
    zero_variance_eps: float = 1e-12

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
