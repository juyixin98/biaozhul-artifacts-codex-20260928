"""HTTP request and response contracts."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator


class FullMatrix(BaseModel):
    type: Literal["full"] = "full"
    data: list[list[float]]


class DiagonalMatrix(BaseModel):
    type: Literal["diag"]
    diagonal: list[float]


MatrixInput = Union[list[list[float]], FullMatrix, DiagonalMatrix]


class WorkerInput(BaseModel):
    id: str = Field(min_length=1)
    P: MatrixInput
    q: list[float]
    r: float = 0.0
    sample_count: int = Field(default=1, gt=0)


class RhoChange(BaseModel):
    iteration: int = Field(ge=0)
    rho: float = Field(gt=0)


class RunCreate(BaseModel):
    name: str = Field(default="admm-run", min_length=1, max_length=128)
    dimension: int = Field(gt=0)
    workers: list[WorkerInput] = Field(min_length=2, max_length=2)
    initial_rho: float = Field(default=1.0, gt=0)
    rho_schedule: list[RhoChange] = Field(default_factory=list)
    initial_z: list[float] | None = None
    eps_abs: float = Field(default=1e-6, gt=0)
    eps_rel: float = Field(default=1e-6, gt=0)
    max_iterations: int = Field(default=100, gt=0, le=1000)
    deadline_ms: int = Field(default=1000, gt=0, le=60000)

    @field_validator("workers")
    @classmethod
    def unique_worker_ids(cls, workers: list[WorkerInput]) -> list[WorkerInput]:
        if len({worker.id for worker in workers}) != len(workers):
            raise ValueError("worker ids must be unique")
        return workers

    @field_validator("rho_schedule")
    @classmethod
    def unique_rho_iterations(cls, schedule: list[RhoChange]) -> list[RhoChange]:
        if len({item.iteration for item in schedule}) != len(schedule):
            raise ValueError("rho_schedule iterations must be unique")
        return schedule


class SubmissionCreate(BaseModel):
    round_token: str
    rho: float = Field(gt=0)
    z_pre: list[float]
    u_pre: list[float]
    x: list[float]
    objective: float
    stationarity_residual: float = Field(ge=0.0)
    solve_seconds: float = Field(default=0.0, ge=0.0)


class TimeoutRequest(BaseModel):
    reason: str = Field(default="worker_deadline_exceeded", max_length=200)


class ResumeRequest(BaseModel):
    deadline_ms: int | None = Field(default=None, gt=0, le=60000)
    reason: str = Field(default="resumed_after_timeout", max_length=200)
