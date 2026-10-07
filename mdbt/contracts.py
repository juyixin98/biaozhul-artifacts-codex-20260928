"""Data contracts passed across module boundaries.

These dataclasses (plus :mod:`mdbt.errors`) are the only shared vocabulary
between scheduler, executor, reducer, storage and API layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ErrorCategory


@dataclass(frozen=True)
class Command:
    """One operation chosen against one resource."""

    resource_id: str
    action: str

    def to_dict(self) -> dict:
        return {"resource_id": self.resource_id, "action": self.action}

    @classmethod
    def from_dict(cls, raw: dict) -> "Command":
        return cls(resource_id=str(raw["resource_id"]), action=str(raw["action"]))


@dataclass(frozen=True)
class Observation:
    """Normalised SUT response (the single source of truth for the model)."""

    accepted: bool
    status_code: int
    state: str | None
    resource_exists: bool
    category: ErrorCategory | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "status_code": self.status_code,
            "state": self.state,
            "resource_exists": self.resource_exists,
            "category": None if self.category is None else self.category.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Verdict:
    kind: str  # "consistent" | "discrepancy"
    category: ErrorCategory | None
    rationale: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "category": None if self.category is None else self.category.value,
            "rationale": self.rationale,
        }


@dataclass
class StepRecord:
    seq: int
    command: Command
    decision: dict
    model_state_before: str
    expected: dict
    observation: dict
    model_state_after: str
    resynced: bool
    verdict: dict

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "command": self.command.to_dict(),
            "decision": self.decision,
            "model_state_before": self.model_state_before,
            "expected": self.expected,
            "observation": self.observation,
            "model_state_after": self.model_state_after,
            "resynced": self.resynced,
            "verdict": self.verdict,
        }


@dataclass
class Finding:
    category: ErrorCategory
    title: str
    rationale: str
    trigger_seq: int
    command: Command
    expected: dict
    observation: dict
    reduced: bool = False
    reduced_length: int | None = None
    reduction_attempted: bool = False
    reduction_log: list = field(default_factory=list)
    replay_script: str = ""
    replay_path: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "title": self.title,
            "rationale": self.rationale,
            "trigger_seq": self.trigger_seq,
            "command": self.command.to_dict(),
            "expected": self.expected,
            "observation": self.observation,
            "reduced": self.reduced,
            "reduced_length": self.reduced_length,
            "reduction_attempted": self.reduction_attempted,
            "reduction_log": self.reduction_log,
            "replay_script": self.replay_script,
            "replay_path": self.replay_path,
        }
