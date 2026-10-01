"""Pydantic request models exposed by the HTTP boundary."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator
from typing import Literal


Method = Literal["empirical", "ledoit_wolf", "ogk"]


class EstimationRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    feature_names: list[str] | None = Field(default=None, min_length=1)
    rows: list[list[float | None]] = Field(min_length=2)
    methods: list[Method] = Field(default_factory=lambda: ["empirical", "ledoit_wolf", "ogk"])

    @field_validator("rows")
    @classmethod
    def validate_rows(cls, rows: list[list[float | None]]) -> list[list[float | None]]:
        if not rows:
            raise ValueError("at least two rows are required")
        width = len(rows[0])
        if width < 1:
            raise ValueError("at least one column is required")
        for index, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"row {index} has {len(row)} columns; expected {width}")
            for column, value in enumerate(row):
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"row {index}, column {column} must be finite or null")
        return rows

    @field_validator("methods")
    @classmethod
    def validate_methods(cls, methods: list[Method]) -> list[Method]:
        if not methods:
            raise ValueError("at least one method is required")
        return list(dict.fromkeys(methods))
