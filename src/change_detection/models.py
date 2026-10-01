from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    mean: float
    cost: float


@dataclass(frozen=True)
class SegmentationResult:
    objective: float
    sse: float
    penalty_per_segment: float
    total_penalty: float
    segment_count: int
    breakpoints: tuple[int, ...]
    segments: tuple[Segment, ...]
    solution_mode: str
    transition_count: int
    pruned_transitions: int
    verification_mode: str
    tie_rule: str = "lower_objective;then_fewer_segments;then_lexicographic_breakpoints"


@dataclass(frozen=True)
class DpState:
    objective: float
    sse: float
    previous: int
    breakpoints: tuple[int, ...]
    segment_count: int
    transition_count: int
    pruned_transitions: int


def segment_cost_key(objective: float, segment_count: int, breakpoints: tuple[int, ...]) -> tuple[float, int, tuple[int, ...]]:
    return objective, segment_count, breakpoints

