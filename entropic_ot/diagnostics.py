"""Diagnostic evidence used by both the solver and service layer."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


def new_run_id() -> str:
    """Return a globally replayable run identifier."""

    return uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class IterationEvent:
    iteration: int
    row_marginal_error: float
    column_marginal_error: float
    combined_error_ratio: float
    finite_log_plan: bool


@dataclass(slots=True)
class EventCollector:
    """Keep the first and most recent events without storing a huge trajectory."""

    keep_first: int = 10
    keep_last: int = 20
    events: list[IterationEvent] = field(default_factory=list)

    def add(self, event: IterationEvent) -> None:
        self.events.append(event)
        overflow = len(self.events) - (self.keep_first + self.keep_last)
        if overflow > 0:
            del self.events[self.keep_first : self.keep_first + overflow]


@dataclass(slots=True)
class Diagnostics:
    run_id: str
    status: str
    stopped_reason: str
    feasible: bool
    converged: bool
    iterations: int
    max_iterations: int
    active_shape: tuple[int, int]
    row_marginal_error: float
    column_marginal_error: float
    combined_error_ratio: float
    row_tolerance: float
    column_tolerance: float
    events: list[IterationEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Convert dataclasses to primitives for JSON logging and API responses.

        return {
            "run_id": self.run_id,
            "status": self.status,
            "stopped_reason": self.stopped_reason,
            "feasible": self.feasible,
            "converged": self.converged,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "active_shape": list(self.active_shape),
            "row_marginal_error": self.row_marginal_error,
            "column_marginal_error": self.column_marginal_error,
            "combined_error_ratio": self.combined_error_ratio,
            "row_tolerance": self.row_tolerance,
            "column_tolerance": self.column_tolerance,
            "events": [
                {
                    "iteration": event.iteration,
                    "row_marginal_error": event.row_marginal_error,
                    "column_marginal_error": event.column_marginal_error,
                    "combined_error_ratio": event.combined_error_ratio,
                    "finite_log_plan": event.finite_log_plan,
                }
                for event in self.events
            ],
        }
