"""Pydantic request/response models for the HTTP boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CorpusIn(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=64)
    text: str = Field(..., min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)


class CorpusOut(BaseModel):
    corpus_id: str
    sentences: int
    tokens: int


class TrainIn(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=64)
    corpus_id: str = Field(..., min_length=1, max_length=64)
    record_events: int = Field(default=0, ge=0, le=100)
    config: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    run_id: str
    corpus_id: str
    status: Literal["PENDING", "COMPLETED", "FAILED"]
    config: dict[str, Any]
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    stats: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str
    updated_at: str


class TrainOut(BaseModel):
    run_id: str
    corpus_id: str
    status: Literal["COMPLETED"]
    config: dict[str, Any]
    stats: dict[str, Any]
    artifact: dict[str, Any]
    trace: list[dict[str, Any]] | None = None


class VectorOut(BaseModel):
    run_id: str
    word: str
    word_id: int
    vector: list[float]
    norm: float


class SimilarityIn(BaseModel):
    word_a: str = Field(..., min_length=1)
    word_b: str = Field(..., min_length=1)


class SimilarityOut(BaseModel):
    run_id: str
    word_a: str
    word_b: str
    word_id_a: int
    word_id_b: int
    similarity: float


class NeighboursIn(BaseModel):
    word: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class Neighbour(BaseModel):
    word: str
    word_id: int
    similarity: float


class NeighboursOut(BaseModel):
    run_id: str
    word: str
    word_id: int
    neighbours: list[Neighbour]


class Pair(BaseModel):
    word_a: str
    word_b: str
    label: Literal[0, 1]


class EvaluateIn(BaseModel):
    pairs: list[Pair] = Field(..., min_length=1)


class EvaluateOut(BaseModel):
    run_id: str
    pair_count: int
    threshold: float
    agreement: float
    agreements: int
    results: list[dict[str, Any]]
    note: str
