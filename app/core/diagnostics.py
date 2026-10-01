from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FitStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    CONVERGED_COST_TOLERANCE = "CONVERGED_COST_TOLERANCE"
    CONVERGED_STEP_SIZE = "CONVERGED_STEP_SIZE"
    NON_CONVERGED_MAX_ITERATIONS = "NON_CONVERGED_MAX_ITERATIONS"
    NON_CONVERGED_TRUST_REGION = "NON_CONVERGED_TRUST_REGION"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    INVALID_INITIAL_RESIDUAL = "INVALID_INITIAL_RESIDUAL"
    INVALID_TRIAL_VALUE = "INVALID_TRIAL_VALUE"
    DERIVATIVE_MISMATCH = "DERIVATIVE_MISMATCH"
    INVALID_EXPRESSION = "INVALID_EXPRESSION"
    INVALID_INPUT = "INVALID_INPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


SUCCESS_STATUSES = {
    FitStatus.OPTIMAL,
    FitStatus.CONVERGED_COST_TOLERANCE,
    FitStatus.CONVERGED_STEP_SIZE,
}


class FitError(ValueError):
    def __init__(
        self,
        status: FitStatus,
        message: str,
        *,
        run_id: str | None = None,
        input_fingerprint: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.run_id = run_id
        self.input_fingerprint = input_fingerprint


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    accepted: bool
    cost: float
    actual_reduction: float | None
    predicted_reduction: float | None
    reduction_ratio: float | None
    radius: float
    step_norm_scaled: float
    projected_gradient_norm: float
    active_lower: list[str]
    active_upper: list[str]
    bound_hit: list[str]
    judgment: str


@dataclass(slots=True)
class DerivativeCheck:
    passed: bool
    max_relative_error: float
    worst_parameter: str | None
    worst_observation: int | None
    relative_step: float


@dataclass(slots=True)
class Identifiability:
    rank: int
    parameter_count: int
    identifiable: bool
    singular_values: list[float]
    condition_number: float | None
    parameter_correlation: list[list[float | None]]


@dataclass(slots=True)
class FitResult:
    status: FitStatus
    success: bool
    message: str
    reason: str
    parameter_names: list[str]
    x: np.ndarray
    residuals: np.ndarray
    cost: float
    jacobian: np.ndarray
    projected_gradient_norm: float
    stationarity_type: str
    active_lower: np.ndarray
    active_upper: np.ndarray
    scales: np.ndarray
    iterations: list[IterationRecord] = field(default_factory=list)
    nfev: int = 0
    njev: int = 0
    invalid_evaluations: int = 0
    final_radius: float = 1.0
    identifiability: Identifiability | None = None
    derivative_check: DerivativeCheck | None = None

    def parameter_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.parameter_names, self.x)}

    def active_map(self) -> tuple[dict[str, bool], dict[str, bool]]:
        lower = {
            name: bool(active)
            for name, active in zip(self.parameter_names, self.active_lower)
        }
        upper = {
            name: bool(active)
            for name, active in zip(self.parameter_names, self.active_upper)
        }
        return lower, upper


def finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FitError(FitStatus.INVALID_INPUT, f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise FitError(FitStatus.INVALID_INPUT, f"{name} must be finite")
    return number
