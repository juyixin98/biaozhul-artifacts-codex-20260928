from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .validation import CleanPoint


@dataclass(frozen=True, slots=True)
class Block:
    min_score: float
    max_score: float
    weight: float
    target_sum: float
    target_square_sum: float
    score_indexes: tuple[int, ...]

    @property
    def probability(self) -> float:
        return self.target_sum / self.weight

    @property
    def weighted_sse(self) -> float:
        return self.target_square_sum - (self.target_sum * self.target_sum) / self.weight

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["probability"] = self.probability
        data["weighted_sse"] = self.weighted_sse
        return data


@dataclass(frozen=True, slots=True)
class MergeEvent:
    sequence: int
    lower_score: float
    upper_score: float
    left_weight: float
    right_weight: float
    left_probability: float
    right_probability: float
    merged_probability: float
    weighted_loss_delta: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PAVResult:
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    blocks: tuple[Block, ...]
    merge_events: tuple[MergeEvent, ...]
    weighted_sse: float
    total_weight: float
    grouped_points: tuple[CleanPoint, ...] = field(default_factory=tuple)


def _merge(left: Block, right: Block, *, previous_loss: float, sequence: int) -> tuple[Block, MergeEvent]:
    merged = Block(
        min_score=left.min_score,
        max_score=right.max_score,
        weight=left.weight + right.weight,
        target_sum=left.target_sum + right.target_sum,
        target_square_sum=left.target_square_sum + right.target_square_sum,
        score_indexes=left.score_indexes + right.score_indexes,
    )
    event = MergeEvent(
        sequence=sequence,
        lower_score=merged.min_score,
        upper_score=merged.max_score,
        left_weight=left.weight,
        right_weight=right.weight,
        left_probability=left.probability,
        right_probability=right.probability,
        merged_probability=merged.probability,
        weighted_loss_delta=merged.weighted_sse - previous_loss,
    )
    return merged, event


def aggregate_equal_scores(points: list[CleanPoint]) -> list[CleanPoint]:
    ordered = sorted(points, key=lambda point: (point.score, point.record_id or ""))
    grouped: list[CleanPoint] = []
    index = 0
    while index < len(ordered):
        score = ordered[index].score
        weight = 0.0
        target_sum = 0.0
        while index < len(ordered) and ordered[index].score == score:
            current = ordered[index]
            weight += current.weight
            target_sum += current.weight * current.target
            index += 1
        grouped.append(CleanPoint(score=score, target=target_sum / weight, weight=weight, record_id=None))
        assert weight > 0.0
    return grouped


def fit_pav(points: list[CleanPoint]) -> PAVResult:
    """Fit weighted isotonic regression with the pool-adjacent-violators algorithm."""
    grouped = aggregate_equal_scores(points)
    stack: list[Block] = []
    events: list[MergeEvent] = []

    for score_index, point in enumerate(grouped):
        target_sum = point.weight * point.target
        current = Block(
            min_score=point.score,
            max_score=point.score,
            weight=point.weight,
            target_sum=target_sum,
            target_square_sum=target_sum * point.target,
            score_indexes=(score_index,),
        )
        while stack and stack[-1].probability > current.probability:
            left = stack.pop()
            previous_loss = left.weighted_sse + current.weighted_sse
            current, event = _merge(left, current, previous_loss=previous_loss, sequence=len(events) + 1)
            events.append(event)
        stack.append(current)

    return PAVResult(
        scores=tuple(block.min_score for block in stack),
        probabilities=tuple(block.probability for block in stack),
        blocks=tuple(stack),
        merge_events=tuple(events),
        weighted_sse=sum(block.weighted_sse for block in stack),
        total_weight=sum(block.weight for block in stack),
        grouped_points=tuple(grouped),
    )
