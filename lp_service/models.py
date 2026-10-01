"""HTTP request and response schemas."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ObjectiveSense = Literal["min", "max"]
ConstraintSense = Literal["eq", "le", "ge"]


class Bound(BaseModel):
    lower: float | None = Field(default=None, description="null means -infinity")
    upper: float | None = Field(default=None, description="null means +infinity")

    @model_validator(mode="after")
    def validate_bound(self) -> "Bound":
        if self.lower is not None and not _is_finite(self.lower):
            raise ValueError("bound lower must be finite or null")
        if self.upper is not None and not _is_finite(self.upper):
            raise ValueError("bound upper must be finite or null")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("bound lower cannot exceed upper")
        return self


class Constraint(BaseModel):
    name: str | None = None
    sense: ConstraintSense
    rhs: float
    # A list is dense; a dict is useful for sparse synthetic fixtures.
    coefficients: dict[str, float] | list[float]

    @field_validator("rhs")
    @classmethod
    def validate_rhs(cls, value: float) -> float:
        if not _is_finite(value):
            raise ValueError("constraint rhs must be finite")
        return value

    @model_validator(mode="after")
    def validate_coefficients(self) -> "Constraint":
        values = self.coefficients.values() if isinstance(self.coefficients, dict) else self.coefficients
        if not all(_is_finite(value) for value in values):
            raise ValueError("constraint coefficients must be finite")
        return self


class LPRequest(BaseModel):
    objective: list[float] = Field(..., min_length=1)
    sense: ObjectiveSense = "min"
    constraints: list[Constraint] = Field(default_factory=list)
    bounds: list[Bound | None] = Field(default_factory=list)

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, values: list[float]) -> list[float]:
        if not all(_is_finite(value) for value in values):
            raise ValueError("objective coefficients must be finite")
        return values

    @model_validator(mode="after")
    def validate_dimensions(self) -> "LPRequest":
        n = len(self.objective)
        if self.bounds and len(self.bounds) != n:
            raise ValueError("bounds length must equal objective length")
        for i, constraint in enumerate(self.constraints):
            if isinstance(constraint.coefficients, list) and len(constraint.coefficients) != n:
                raise ValueError(f"constraint {i} coefficient length must equal objective length")
        return self


class Diagnostic(BaseModel):
    accepted: bool
    checks: dict[str, bool]
    residuals: dict[str, float | list[float]]
    message: str


class LPFResponse(BaseModel):
    status: Literal["optimal", "infeasible", "unbounded", "numerical_uncertain"]
    request_id: str
    objective_value: float | None = None
    solution: list[float] | None = None
    residuals: list[float] | None = None
    bound_violations: list[float] | None = None
    duals: dict[str, list[float]] | None = None
    unbounded_direction: list[float] | None = None
    certificate: dict[str, object] | None = None
    diagnostic: Diagnostic


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
