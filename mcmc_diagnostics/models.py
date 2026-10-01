"""HTTP sample and response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiagnosticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Caller-chosen correlation identifier. Generated when omitted.",
    )
    chains: list[list[float]] = Field(
        ...,
        description="Equal-length chains. chains[m][n] is draw n of chain m.",
    )
    max_acf_lag: int | None = Field(
        default=None,
        ge=0,
        description="Maximum lag included in the explainable ACF report.",
    )
    persist: bool = Field(
        default=True,
        description="Store the versioned result in SQLite. Invalid requests are never stored.",
    )

    @field_validator("chains")
    @classmethod
    def validate_chains(cls, chains: list[list[float]]) -> list[list[float]]:
        if not chains:
            raise ValueError("At least one chain is required; EMPTY_SAMPLES.")
        lengths = {len(chain) for chain in chains}
        if 0 in lengths:
            raise ValueError("Empty chains are not supported; EMPTY_CHAIN.")
        if len(lengths) != 1:
            raise ValueError(
                "All chains must have the same length; EQUAL_LENGTH_REQUIRED."
            )
        return chains


class ErrorResponse(BaseModel):
    request_id: str
    schema_version: str
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    processing_location: str
