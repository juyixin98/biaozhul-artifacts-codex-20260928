"""Strict request parsing and validation.

The parser deliberately consumes raw dictionaries so that common malformed
local fixtures get deterministic, specific error codes rather than a generic
framework validation message.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from app.errors import validation_error
from app.kernel import CandidateInput, Direction, Objective


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise validation_error(
            "INVALID_OBJECT",
            f"field {field} must be an object",
            {"field": field},
        )
    return value


def parse_objectives(raw: Any) -> tuple[Objective, ...]:
    if not isinstance(raw, list) or not raw:
        raise validation_error(
            "INVALID_OBJECTIVES",
            "objectives must be a non-empty list",
        )

    objectives: list[Objective] = []
    names: set[str] = set()
    valid_directions = {"min", "max"}

    for index, item in enumerate(raw):
        raw_objective = _require_object(item, f"objectives[{index}]")
        name = raw_objective.get("name")
        direction = raw_objective.get("direction")

        if not isinstance(name, str) or not name:
            raise validation_error(
                "INVALID_OBJECTIVE_NAME",
                f"objectives[{index}].name must be a non-empty string",
                {"index": index},
            )
        if name in names:
            raise validation_error(
                "DUPLICATE_OBJECTIVE_NAME",
                f"duplicate objective name: {name}",
                {"name": name, "index": index},
            )
        if direction not in valid_directions:
            raise validation_error(
                "INVALID_DIRECTION",
                f"objectives[{index}].direction must be one of {sorted(valid_directions)}",
                {"name": name, "direction": direction},
            )

        names.add(name)
        objectives.append(Objective(name=name, direction=direction))

    return tuple(objectives)


def _parse_number(candidate_id: str, objective_name: str, value: Any) -> float:
    # bool is a subclass of int in Python, but a Boolean is not a declared
    # objective number in this API.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise validation_error(
            "NON_NUMERIC_OBJECTIVE",
            f"candidate {candidate_id} objective {objective_name} must be a number",
            {"candidate_id": candidate_id, "objective": objective_name, "value": repr(value)},
        )
    number = float(value)
    if not math.isfinite(number):
        raise validation_error(
            "NON_FINITE_OBJECTIVE",
            f"candidate {candidate_id} objective {objective_name} must be finite",
            {"candidate_id": candidate_id, "objective": objective_name, "value": number},
        )
    return number


def parse_candidates(
    raw: Any,
    objectives: Iterable[Objective],
    *,
    required: bool = True,
) -> tuple[CandidateInput, ...]:
    if not isinstance(raw, list):
        raise validation_error(
            "INVALID_CANDIDATES",
            "candidates must be a list",
        )
    if required and not raw:
        raise validation_error(
            "INVALID_CANDIDATES",
            "candidates must be a non-empty list",
        )

    objective_tuple = tuple(objectives)
    objective_names = tuple(objective.name for objective in objective_tuple)
    parsed: list[CandidateInput] = []
    ids: set[str] = set()

    for index, item in enumerate(raw):
        raw_candidate = _require_object(item, f"candidates[{index}]")
        candidate_id = raw_candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise validation_error(
                "INVALID_CANDIDATE_ID",
                f"candidates[{index}].id must be a non-empty string",
                {"index": index},
            )
        if candidate_id in ids:
            raise validation_error(
                "DUPLICATE_CANDIDATE_ID",
                f"duplicate candidate id in request: {candidate_id}",
                {"candidate_id": candidate_id},
            )

        values = raw_candidate.get("values")
        if not isinstance(values, dict):
            raise validation_error(
                "INVALID_VALUES",
                f"candidate {candidate_id} values must be an object",
                {"candidate_id": candidate_id},
            )

        vector: list[float] = []
        missing = [name for name in objective_names if name not in values]
        if missing:
            raise validation_error(
                "MISSING_OBJECTIVE",
                f"candidate {candidate_id} is missing objective values",
                {"candidate_id": candidate_id, "missing_objectives": missing},
            )

        extra = sorted(set(values) - set(objective_names))
        if extra:
            raise validation_error(
                "UNKNOWN_OBJECTIVE",
                f"candidate {candidate_id} contains undeclared objectives",
                {"candidate_id": candidate_id, "unknown_objectives": extra},
            )

        for objective in objective_tuple:
            vector.append(_parse_number(candidate_id, objective.name, values[objective.name]))

        ids.add(candidate_id)
        parsed.append(CandidateInput(id=candidate_id, vector=tuple(vector)))

    return tuple(parsed)


def parse_tolerance(raw: Any) -> float:
    if raw is None:
        raw = 0.0
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise validation_error(
            "INVALID_TOLERANCE",
            "tolerance must be a non-negative number",
            {"value": repr(raw)},
        )
    tolerance = float(raw)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise validation_error(
            "INVALID_TOLERANCE",
            "tolerance must be a finite non-negative number",
            {"value": tolerance},
        )
    return tolerance


def parse_problem(
    payload: dict[str, Any],
    *,
    candidates_required: bool = True,
) -> tuple[tuple[Objective, ...], tuple[CandidateInput, ...], float]:
    objectives = parse_objectives(payload.get("objectives"))
    candidates = parse_candidates(
        payload.get("candidates"),
        objectives,
        required=candidates_required,
    )
    tolerance = parse_tolerance(payload.get("tolerance"))
    return objectives, candidates, tolerance

