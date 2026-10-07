from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TrainPoint(StrictModel):
    score: float
    target: int
    weight: float | None = None
    record_id: str | None = None


class TrainRequest(StrictModel):
    score_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    records: list[TrainPoint] = Field(min_length=1)
    extrapolation: Literal["clip", "raise"] = "clip"


class PredictPoint(StrictModel):
    score: float
    record_id: str | None = None


class PredictRequest(StrictModel):
    score_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    records: list[PredictPoint] = Field(min_length=1)
    extrapolation: Literal["clip", "raise"] | None = None


class EvaluatePoint(StrictModel):
    score: float
    target: int
    weight: float | None = None
    record_id: str | None = None


class EvaluateRequest(StrictModel):
    score_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    records: list[EvaluatePoint] = Field(min_length=1)


class Step(BaseModel):
    module: str
    version: str
    description: str


class ResponseEnvelope(BaseModel):
    request_id: str
    status: Literal["succeeded", "failed"]
    result: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
