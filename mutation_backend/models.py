"""Pydantic input and response models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MutationOperator(StrEnum):
    COMPARISON = "comparison"
    ARITHMETIC = "arithmetic"
    CONDITIONAL_INVERT = "conditional_invert"
    CONDITION_ERASE = "condition_erase"


class MutantFate(StrEnum):
    PENDING = "pending"
    SYNTAX_INVALID = "syntax_invalid"
    STARTUP_FAILED = "startup_failed"
    KILLED = "killed"
    SURVIVED = "survived"
    TIMEOUT = "timeout"
    EQUIVALENT_UNDETERMINED = "equivalent_undetermined"
    RUNNER_ERROR = "runner_error"


class RunStatus(StrEnum):
    PENDING = "pending"
    BASELINE_FAILED = "baseline_failed"
    BASELINE_TIMEOUT = "baseline_timeout"
    COMPLETED = "completed"
    FAILED = "failed"


class MutationRequest(BaseModel):
    request_id: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{3,63}$")
    project_path: Path = Field(..., description="Absolute path to a local Python project.")
    test_command: list[str] = Field(default_factory=lambda: ["python3", "-m", "pytest", "-q"])
    source_paths: list[str] = Field(default_factory=lambda: ["**/*.py"])
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["**/tests/**", "**/test_*.py", "**/__pycache__/**"]
    )
    operators: list[MutationOperator] = Field(
        default_factory=lambda: [
            MutationOperator.COMPARISON,
            MutationOperator.ARITHMETIC,
            MutationOperator.CONDITIONAL_INVERT,
        ]
    )
    baseline_timeout_seconds: float = Field(10, gt=0)
    mutant_timeout_seconds: float = Field(1, gt=0)

    @field_validator("project_path")
    @classmethod
    def project_must_be_absolute(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"project path does not exist or is not a directory: {resolved}")
        return resolved

    @field_validator("test_command")
    @classmethod
    def command_must_be_argv(cls, value: list[str]) -> list[str]:
        if not value or not all(item.strip() for item in value):
            raise ValueError("test_command must be a non-empty argv list")
        return value


class SourceLocation(BaseModel):
    path: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    node_type: str
    original_snippet: str


class MutantResult(BaseModel):
    mutant_id: str
    operator_family: MutationOperator
    before: str
    after: str
    location: SourceLocation
    fate: MutantFate
    determination: Literal["determined", "undetermined"]
    exit_code: int | None = None
    duration_seconds: float | None = None
    reason: str
    failure_category: str | None = None
    stderr_tail: str | None = None
    equivalent_basis: str | None = None
    workspace: str | None = None


class SourceFileEvidence(BaseModel):
    path: str
    sha256: str
    size_bytes: int


class RunEvent(BaseModel):
    sequence: int
    timestamp: str
    step: str
    status: str
    detail: str
    location: str | None = None
    engine_version: str


class MutationReport(BaseModel):
    request_id: str
    status: RunStatus
    project_path: str
    engine_version: str
    baseline: dict[str, Any]
    timeouts: dict[str, float]
    summary: dict[str, int]
    uncertainties: list[MutantResult]
    failures: list[MutantResult]
    mutants: list[MutantResult]
    source_before: dict[str, Any]
    source_after: dict[str, Any]
    source_unchanged: bool
    events: list[RunEvent]
    created_at: str | None = None
    completed_at: str | None = None
