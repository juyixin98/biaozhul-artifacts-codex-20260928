from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Mapping

from .errors import ValidationFailure


@dataclass(frozen=True, slots=True)
class CleanPoint:
    score: float
    target: int
    weight: float
    record_id: str | None


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    message: str
    index: int
    record_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_identifier(record: Mapping[str, Any]) -> str | None:
    value = record.get("record_id")
    return value if isinstance(value, str) else None


def _issue(code: str, message: str, index: int, record_id: str | None) -> Issue:
    return Issue(code=code, message=message, index=index, record_id=record_id)


def _finite_number(name: str, value: Any, issues: list[Issue], index: int, record_id: str | None) -> float | None:
    if value is None or not isinstance(value, Real) or isinstance(value, bool):
        issues.append(_issue(f"invalid_{name}", f"{name} must be a finite number", index, record_id))
        return None
    converted = float(value)
    if not math.isfinite(converted):
        issues.append(_issue(f"non_finite_{name}", f"{name} must be finite", index, record_id))
        return None
    return converted


def _binary_target(value: Any, issues: list[Issue], index: int, record_id: str | None) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        issues.append(_issue("invalid_target", "target must be binary integer 0 or 1", index, record_id))
        return None
    if value not in (0, 1):
        issues.append(_issue("non_binary_target", "target must be binary integer 0 or 1", index, record_id))
        return None
    return value


def _clean_record(
    record: Any,
    index: int,
    *,
    require_target: bool,
    issues: list[Issue],
) -> CleanPoint | None:
    if not isinstance(record, Mapping):
        issues.append(_issue("invalid_record", "record must be an object", index, None))
        return None

    record_id = _record_identifier(record)
    if "record_id" in record and record_id is None:
        issues.append(_issue("invalid_record_id", "record_id must be a string when provided", index, None))

    score = _finite_number("score", record.get("score"), issues, index, record_id)

    target = 0
    target_valid = True
    if require_target:
        if "target" not in record or record.get("target") is None:
            issues.append(_issue("missing_target", "target is required and cannot be null", index, record_id))
            target_valid = False
        else:
            parsed_target = _binary_target(record.get("target"), issues, index, record_id)
            if parsed_target is None:
                target_valid = False
            else:
                target = parsed_target
    elif "target" in record and record.get("target") is not None:
        parsed_target = _binary_target(record.get("target"), issues, index, record_id)
        if parsed_target is None:
            target_valid = False
        else:
            target = parsed_target

    weight_value: Any = 1.0
    if "weight" in record and record.get("weight") is not None:
        weight_value = record.get("weight")
    weight = _finite_number("weight", weight_value, issues, index, record_id)
    if weight is not None and weight < 0.0:
        issues.append(_issue("negative_weight", "weight must be non-negative", index, record_id))
        weight = None

    if score is None or weight is None or not target_valid:
        return None
    return CleanPoint(score=score, target=target, weight=weight, record_id=record_id)


def validate_records(records: Iterable[Mapping[str, Any]], *, require_target: bool = True) -> tuple[list[CleanPoint], list[Issue]]:
    points: list[CleanPoint] = []
    issues: list[Issue] = []
    materialized = list(records)
    if not materialized:
        issues.append(Issue(code="empty_records", message="at least one record is required", index=-1))

    for index, record in enumerate(materialized):
        point = _clean_record(record, index, require_target=require_target, issues=issues)
        if point is not None:
            points.append(point)

    errors = [issue for issue in issues if issue.code != "zero_weight"]
    if errors:
        raise ValidationFailure("input validation failed", details=[issue.as_dict() for issue in errors])

    warnings: list[Issue] = []
    positive_points: list[CleanPoint] = []
    for point in points:
        if point.weight == 0.0:
            warnings.append(Issue("zero_weight", "zero-weight record is excluded from fitting", -1, point.record_id))
        else:
            positive_points.append(point)

    if require_target and not positive_points:
        raise ValidationFailure(
            "no positive-weight training records",
            details=[Issue("no_effective_records", "training requires at least one positive-weight record", -1).as_dict()],
        )

    return positive_points, warnings
