"""Pydantic HTTP request/response contracts."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    corpus: list[list[str]] = Field(..., min_length=1)
    seed: int = Field(1729, ge=0, le=2**32 - 1)
    epochs: int = Field(5, ge=1, le=1000)
    window: int = Field(2, ge=1, le=32)
    negatives: int = Field(5, ge=1, le=100)
    dim: int = Field(16, ge=2, le=512)
    learning_rate: float = Field(0.025, gt=0.0, le=1.0)
    subsample_t: float = Field(1e-3, gt=0.0, le=1.0)
    min_count: int = Field(1, ge=1)
    init_range: float = Field(0.5, gt=0.0, le=10.0)


class TrainResponse(BaseModel):
    run_id: str
    status: str
    vocab_size: int
    dim: int
    final_loss: float
    corpus_fingerprint: str


class VectorResponse(BaseModel):
    run_id: str
    token: str
    vector: list[float]


class SimilarityResponse(BaseModel):
    run_id: str
    token_a: str
    token_b: str
    similarity: float


class Neighbor(BaseModel):
    token: str
    similarity: float


class NeighborsResponse(BaseModel):
    run_id: str
    token: str
    neighbors: list[Neighbor]


class EvaluationRequest(BaseModel):
    pairs: list[tuple[str, str]] = Field(..., min_length=1)


class EvaluationResponse(BaseModel):
    evaluation_id: str
    run_id: str
    metrics: dict


class RunResponse(BaseModel):
    run_id: str
    status: str
    config: dict
    corpus_fingerprint: str
    final_loss: float | None
    error_category: str | None
    error_message: str | None


class EventsResponse(BaseModel):
    run_id: str
    event_count: int
    events: list[dict]
