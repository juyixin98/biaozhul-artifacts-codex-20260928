"""Input model, verdicts and job specification."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    """Failure categories distinguished by the oracle."""

    TARGET = "target"  # the target error reproduced
    OTHER = "other"  # a different error
    PASS = "pass"  # success
    INDETERMINATE = "indeterminate"  # cannot decide (e.g. timeout)


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # interrupted, resumable from checkpoint
    DONE = "done"
    FAILED = "failed"


class StructuredInput(BaseModel):
    """Structured input with declared basic syntax constraints:

    - top-level object with a ``records`` array;
    - every record is an object carrying a unique non-empty string ``id``.

    The only reduction operator is whole-record removal, so every
    candidate the reducer produces satisfies these constraints by
    construction (subsets of a valid record list stay valid).
    """

    records: list[dict[str, Any]]

    @field_validator("records")
    @classmethod
    def _check_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"record {i} is not an object")
            rid = record.get("id")
            if not isinstance(rid, str) or not rid:
                raise ValueError(f"record {i} lacks a non-empty string 'id'")
            if rid in seen:
                raise ValueError(f"duplicate record id {rid!r}")
            seen.add(rid)
        return records

    def canonical_json(self) -> str:
        return json.dumps({"records": self.records}, sort_keys=True, separators=(",", ":"))

    def subset(self, ids: list[str]) -> "StructuredInput":
        keep = set(ids)
        return StructuredInput(records=[r for r in self.records if r["id"] in keep])

    @property
    def ids(self) -> list[str]:
        return [r["id"] for r in self.records]


class ClassifierConfig(BaseModel):
    """How raw execution outcomes map to verdicts."""

    target_exit_codes: list[int] = Field(default_factory=lambda: [3])
    pass_exit_codes: list[int] = Field(default_factory=lambda: [0])
    target_stderr_pattern: str | None = None  # optional regex that must also match
    timeout_seconds: float = 5.0


class JobSpec(BaseModel):
    """Everything needed to reproduce and reduce one failure."""

    program: list[str]  # argv, e.g. ["python3", "fixtures/programs/sum_threshold_crash.py"]
    input: StructuredInput
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    env: dict[str, str] = Field(default_factory=dict)  # env overrides, part of cache key
    sensitive_fields: list[str] = Field(default_factory=lambda: ["secret", "token", "password"])
