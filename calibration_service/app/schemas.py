"""Pydantic request/response schemas for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Dataset(BaseModel):
    scores: list[float] = Field(min_length=1)
    labels: list[float] = Field(min_length=1)
    weights: list[float] | None = None  # defaults to all-ones

    @field_validator("labels")
    @classmethod
    def _same_length(cls, v, info):
        scores = info.data.get("scores")
        if scores is not None and len(v) != len(scores):
            raise ValueError("labels and scores must have equal length")
        return v


class FitRequest(BaseModel):
    model_id: str = Field(min_length=1)
    dataset: Dataset
    score_field: str | None = None  # defaults to configured score field


class PredictRequest(BaseModel):
    model_id: str
    scores: list[float] = Field(min_length=1)
    version: int | None = None


class EvaluateRequest(BaseModel):
    model_id: str
    dataset: Dataset  # held-out evaluation set, separate from training data
    version: int | None = None


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    category: str  # validation_error | not_found | not_fitted | internal
    detail: list[int] | str | None = None
