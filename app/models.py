"""Pydantic input/output models for the mutation testing API."""
from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


class MutantOutcome(str, enum.Enum):
    """Terminal fate of a single mutant."""

    INVALID_SYNTAX = "invalid_syntax"      # mutated source does not compile
    STARTUP_FAILURE = "startup_failure"    # test collection/import crashed
    KILLED = "killed"                      # test suite failed on the mutant
    SURVIVED = "survived"                  # test suite passed on the mutant
    TIMEOUT = "timeout"                    # mutant run exceeded its timeout
    UNDETERMINED = "undetermined"          # equivalent mutant; no verdict


class RunStatus(str, enum.Enum):
    """Terminal state of a whole mutation run."""

    COMPLETED = "completed"
    BASELINE_FAILED = "baseline_failed"    # unmutated suite already red
    BASELINE_TIMEOUT = "baseline_timeout"  # unmutated suite exceeded timeout
    INVALID_SOURCE = "invalid_source"      # original source does not compile


class MutationOperator(str, enum.Enum):
    COMPARISON = "comparison"        # relational/equality operator swap
    ARITHMETIC = "arithmetic"        # binary arithmetic operator swap
    CONDITION_NEGATION = "condition_negation"  # negate if/while tests


class RunRequest(BaseModel):
    """Input model for starting a mutation run."""

    source_dir: str = Field(
        ..., min_length=1,
        description="Directory with the Python package under test",
    )
    test_dir: Optional[str] = Field(
        None,
        description="Test directory relative to source_dir; default 'tests'",
    )
    timeout_seconds: float = Field(
        30.0, gt=0, le=600, description="Per-mutant test timeout"
    )
    operators: list[MutationOperator] = Field(
        default_factory=lambda: list(MutationOperator),
        description="Mutation operators to apply",
    )
    requester: str = Field(
        "anonymous", description="Identity of the caller, echoed into logs"
    )


class MutantRecord(BaseModel):
    mutant_id: str
    run_id: str
    file_path: str
    line: int
    col: int
    operator: str
    original_segment: str
    mutated_segment: str
    outcome: MutantOutcome
    detail: str = ""


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    requester: str
    source_dir: str
    source_digest: str
    operator_set_version: str
    python_version: str
    created_at: str
    total: int = 0
    killed: int = 0
    survived: int = 0
    timeout: int = 0
    invalid_syntax: int = 0
    startup_failure: int = 0
    undetermined: int = 0
    baseline_note: str = ""
