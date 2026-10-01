"""FastAPI request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class WorkerProblem(BaseModel):
    worker_id: int | None = None
    Q: list[list[float]]
    q: list[float]


class AdaptiveRhoRequest(BaseModel):
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 2.0
    change_rounds: list[int] = Field(default_factory=list)


class ToleranceRequest(BaseModel):
    primal_abs: float = 1e-7
    dual_abs: float = 1e-7
    primal_rel: float = 0.0
    dual_rel: float = 0.0


class ConfigRequest(BaseModel):
    max_iterations: int = 100
    rho: float = 1.0
    tolerances: ToleranceRequest = Field(default_factory=ToleranceRequest)
    adaptive_rho: AdaptiveRhoRequest | None = None


class CreateRunRequest(BaseModel):
    dimension: int | None = None
    workers: list[WorkerProblem]
    config: ConfigRequest = Field(default_factory=ConfigRequest)

    @model_validator(mode="after")
    def validate_workers(self) -> "CreateRunRequest":
        if len(self.workers) != 2:
            raise ValueError("exactly two workers are required")
        return self


class SubmitRequest(BaseModel):
    worker_id: int
    round_number: int
    rho: float
    x: list[float]


class TimeoutRequest(BaseModel):
    missing_worker_ids: list[int] | None = None
    reason: str = "deadline_exceeded"


class ErrorResponse(BaseModel):
    error: dict[str, str | dict]
