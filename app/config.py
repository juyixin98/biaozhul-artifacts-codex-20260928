"""Configuration objects for the solver and HTTP service."""

from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SolverConfig(BaseModel):
    """Numerical settings.

    Trust radii and step tolerances are in normalized parameter coordinates.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_iterations: int = Field(default=60, ge=1, le=1000)
    gradient_tol: float = Field(default=1.0e-8, gt=0.0)
    parameter_tol: float = Field(default=1.0e-12, gt=0.0)
    trust_radius_initial: float = Field(default=1.0, gt=0.0)
    trust_radius_max: float = Field(default=1.0e4, gt=0.0)
    trust_radius_min: float = Field(default=1.0e-14, gt=0.0)
    accept_ratio: float = Field(default=1.0e-4, gt=0.0, lt=1.0)
    good_ratio: float = Field(default=0.25, gt=0.0, lt=1.0)
    excellent_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
    shrink_factor: float = Field(default=0.25, gt=0.0, lt=1.0)
    expand_factor: float = Field(default=2.0, gt=1.0)
    derivative_check: bool = True
    finite_difference_step: float = Field(default=1.4901161193847656e-8, gt=0.0)
    derivative_rtol: float = Field(default=1.0e-5, gt=0.0)
    derivative_atol: float = Field(default=1.0e-7, gt=0.0)
    rank_tolerance: float = Field(default=1.0e-10, gt=0.0)
    boundary_tolerance: float = Field(default=1.0e-10, gt=0.0)

    @field_validator("good_ratio", "excellent_ratio")
    @classmethod
    def _validate_ratios(cls, value: float, info):
        if value <= 1.0e-4:
            raise ValueError(f"{info.field_name} must be meaningfully positive")
        return value


@dataclass(frozen=True)
class ServiceSettings:
    service_name: str = "bounded-nls-trust-region"
    log_level: str = os.getenv("NLS_LOG_LEVEL", "INFO")
    default_run_header: str = "X-Run-ID"
