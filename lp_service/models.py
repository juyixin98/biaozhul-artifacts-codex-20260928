"""External request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Sense = Literal["min", "max"]
Operator = Literal["<=", ">=", "="]
Status = Literal["optimal", "infeasible", "unbounded", "invalid", "numerical_uncertainty"]


class ConstraintIn(BaseModel):
    name: str | None = None
    coefficients: list[float] | dict[str, float]
    op: Operator = "="
    rhs: float = 0.0


class SolveRequest(BaseModel):
    request_id: str | None = Field(default=None, max_length=100)
    objective: list[float]
    sense: Sense = "min"
    constraints: list[ConstraintIn] = Field(default_factory=list)
    variable_names: list[str] | None = None
    bounds: list[tuple[float | None, float | None]] | None = None
    max_iterations: int = Field(default=1000, ge=1, le=1_000_000)
    tolerance: float | None = Field(default=None, gt=0.0, le=0.1)

    @field_validator("bounds", mode="before")
    @classmethod
    def _bounds_to_lists(cls, value: Any) -> Any:
        if value is None:
            return value
        return [list(item) if item is not None else None for item in value]

    @model_validator(mode="after")
    def _validate_shapes(self) -> "SolveRequest":
        n = len(self.objective)
        if n == 0:
            raise ValueError("objective must contain at least one variable")
        if self.variable_names is not None and len(self.variable_names) != n:
            raise ValueError("variable_names length must match objective length")
        if self.bounds is not None and len(self.bounds) != n:
            raise ValueError("bounds length must match objective length")
        if self.variable_names is not None and len(set(self.variable_names)) != n:
            raise ValueError("variable_names must be unique")
        for row, constraint in enumerate(self.constraints):
            constraint.coefficients = self._coeff_vector(constraint.coefficients, n, row)
        return self

    @staticmethod
    def _coeff_vector(value: list[float] | dict[str, float], n: int, row: int) -> list[float]:
        if isinstance(value, dict):
            vector = [0.0] * n
            for key, coefficient in value.items():
                if not isinstance(key, str) or not key.isdigit():
                    raise ValueError(
                        f"constraint[{row}] named coefficient {key!r} requires variable_names"
                    )
                index = int(key)
                if index >= n:
                    raise ValueError(f"constraint[{row}] references variable index {index}")
                vector[index] = float(coefficient)
            return vector
        if len(value) != n:
            raise ValueError(f"constraint[{row}] coefficients length must match objective length")
        return [float(x) for x in value]


class ConstraintDiagnostic(BaseModel):
    name: str
    op: Operator
    rhs: float
    residual: float
    violated: bool


class BoundDiagnostic(BaseModel):
    variable: str
    lower: float | None
    upper: float | None
    value: float
    lower_violation: float
    upper_violation: float


class Evidence(BaseModel):
    primal_residual: float
    bound_violation: float
    objective_gap: float | None = None
    reduced_cost_violation: float | None = None
    dual_sign_violation: float | None = None
    complementary_slack_violation: float | None = None
    ray_step: float | None = None
    farkas_gap: float | None = None


class SolveResponse(BaseModel):
    request_id: str
    status: Status
    message: str
    objective_value: float | None = None
    solution: dict[str, float] | None = None
    unbounded_direction: dict[str, float] | None = None
    constraint_duals: dict[str, float] | None = None
    lower_duals: dict[str, float] | None = None
    upper_duals: dict[str, float] | None = None
    residuals: list[ConstraintDiagnostic] | None = None
    bound_diagnostics: list[BoundDiagnostic] | None = None
    evidence: Evidence
    iterations: int
    pivots: int
    artificial_remaining: int
    redundant_rows: list[int]
    pivot_log: list[dict[str, float | int | str]]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
