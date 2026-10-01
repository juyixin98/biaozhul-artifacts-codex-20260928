"""Version markers and explainable response envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from .. import __version__
from ..versions import DISCLAIMER, KM_ESTIMATOR_VERSION, LOGRANK_VERSION, SCHEMA_VERSION

DataT = TypeVar("DataT", bound=BaseModel)


class ProcessingStep(BaseModel):
    step: str
    status: str
    detail: str


class VersionedResponse(BaseModel, Generic[DataT]):
    request_id: str
    generated_at: datetime
    api_version: str = __version__
    contract_version: str = SCHEMA_VERSION
    processing_location: str
    estimator_versions: list[str]
    processing_steps: list[ProcessingStep] = Field(default_factory=list)
    result: DataT
    disclaimer: str = DISCLAIMER


def resolve_request_id(value: str | None) -> str:
    return value or f"req_{uuid4().hex}"


def build_response(
    *,
    result: DataT,
    request_id: str,
    steps: list[ProcessingStep],
    processing_location: str,
    estimator_versions: list[str] | None = None,
) -> VersionedResponse[DataT]:
    return VersionedResponse[type(result)](
        request_id=request_id,
        generated_at=datetime.now(timezone.utc),
        processing_location=processing_location,
        estimator_versions=estimator_versions
        or [KM_ESTIMATOR_VERSION, LOGRANK_VERSION],
        processing_steps=steps,
        result=result,
    )
