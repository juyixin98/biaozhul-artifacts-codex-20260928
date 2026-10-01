from typing import Literal

from pydantic import BaseModel, Field


class Objective(BaseModel):
    quadratic: list[list[float]] = Field(
        ..., description="Dense symmetric positive-definite Hessian H."
    )
    linear: list[float] | None = Field(default=None, description="Linear objective coefficient vector c.")


class LinearConstraint(BaseModel):
    name: str | None = None
    coefficients: list[float]
    op: Literal["<=", ">=", "=", "=="]
    rhs: float


class ToleranceConfig(BaseModel):
    primal_tol: float | None = Field(default=None, gt=0)
    dual_tol: float | None = Field(default=None, gt=0)
    complementarity_tol: float | None = Field(default=None, gt=0)
    active_tol: float | None = Field(default=None, gt=0)
    rank_tol: float | None = Field(default=None, gt=0)
    step_tol: float | None = Field(default=None, gt=0)
    symmetry_tol: float | None = Field(default=None, gt=0)
    positive_definite_tol: float | None = Field(default=None, gt=0)


class SolveRequest(BaseModel):
    request_id: str | None = Field(default=None, max_length=128)
    objective: Objective
    constraints: list[LinearConstraint] = Field(default_factory=list)
    initial_point: list[float] | None = None
    variable_names: list[str] | None = None
    tolerances: ToleranceConfig | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=100000)


class FeasibilityRequest(BaseModel):
    request_id: str | None = Field(default=None, max_length=128)
    n: int | None = Field(default=None, ge=1)
    constraints: list[LinearConstraint] = Field(default_factory=list)
    initial_point: list[float] | None = None
    variable_names: list[str] | None = None
    tolerances: ToleranceConfig | None = None
