from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import ArtifactError
from .kernel import PAVResult


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    representative_score: float
    block_min_score: float
    block_max_score: float
    lower_threshold: float | None
    upper_threshold: float | None
    probability: float
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    artifact_schema_version: str
    model_id: str
    created_at_utc: str
    score_name: str
    target_name: str
    extrapolation: str
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    thresholds: tuple[float, ...]
    bins: tuple[CalibrationBin, ...]
    blocks: tuple[dict[str, Any], ...]
    merge_events: tuple[dict[str, Any], ...]
    weighted_sse: float
    total_weight: float
    score_min: float
    score_max: float
    input_record_count: int
    grouped_score_count: int
    zero_weight_count: int
    training_fingerprint: str
    score_semantics: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _midpoint(left: float, right: float) -> float:
    return left + (right - left) / 2.0


def _block_right_boundary(pav: PAVResult, block_index: int) -> float:
    next_block = pav.blocks[block_index + 1]
    next_score_index = next_block.score_indexes[0]
    return float(pav.grouped_points[next_score_index - 1].score)


def build_artifact(
    *,
    model_id: str,
    created_at_utc: str,
    score_name: str,
    target_name: str,
    extrapolation: str,
    pav: PAVResult,
    input_record_count: int,
    zero_weight_count: int,
    training_fingerprint: str,
    schema_version: str,
) -> CalibrationArtifact:
    if extrapolation not in {"clip", "raise"}:
        raise ArtifactError("extrapolation must be 'clip' or 'raise'")
    if not pav.scores:
        raise ArtifactError("cannot build artifact without fitted blocks")

    thresholds = tuple(
        _midpoint(_block_right_boundary(pav, index), right.min_score)
        for index, right in enumerate(pav.blocks[1:])
    )
    bins: list[CalibrationBin] = []
    for index, block in enumerate(pav.blocks):
        lower = None if index == 0 else thresholds[index - 1]
        upper = None if index == len(pav.blocks) - 1 else thresholds[index]
        bins.append(
            CalibrationBin(
                representative_score=block.min_score,
                block_min_score=block.min_score,
                block_max_score=block.max_score,
                lower_threshold=lower,
                upper_threshold=upper,
                probability=block.probability,
                weight=block.weight,
            )
        )

    probabilities = tuple(probability for probability in pav.probabilities)
    if any(left > right for left, right in zip(probabilities, probabilities[1:], strict=False)):
        raise ArtifactError("fitted probabilities are not non-decreasing")

    return CalibrationArtifact(
        artifact_schema_version=schema_version,
        model_id=model_id,
        created_at_utc=created_at_utc,
        score_name=score_name,
        target_name=target_name,
        extrapolation=extrapolation,
        scores=pav.scores,
        probabilities=probabilities,
        thresholds=thresholds,
        bins=tuple(bins),
        blocks=tuple(block.as_dict() for block in pav.blocks),
        merge_events=tuple(event.as_dict() for event in pav.merge_events),
        weighted_sse=pav.weighted_sse,
        total_weight=pav.total_weight,
        score_min=pav.scores[0],
        score_max=pav.blocks[-1].max_score,
        input_record_count=input_record_count,
        grouped_score_count=len(pav.scores),
        zero_weight_count=zero_weight_count,
        training_fingerprint=training_fingerprint,
        score_semantics={
            "score_name": score_name,
            "meaning": f"raw input score named {score_name}; larger fitted scores map to non-decreasing probabilities",
            "target_name": target_name,
        },
    )


def artifact_from_dict(data: dict[str, Any]) -> CalibrationArtifact:
    required = {
        "artifact_schema_version",
        "model_id",
        "created_at_utc",
        "score_name",
        "target_name",
        "extrapolation",
        "scores",
        "probabilities",
        "thresholds",
        "bins",
        "blocks",
        "merge_events",
        "weighted_sse",
        "total_weight",
        "score_min",
        "score_max",
        "input_record_count",
        "grouped_score_count",
        "zero_weight_count",
        "training_fingerprint",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ArtifactError(f"artifact missing fields: {', '.join(missing)}")

    bins = tuple(CalibrationBin(**bin_data) for bin_data in data["bins"])
    values = dict(data)
    values["scores"] = tuple(values["scores"])
    values["probabilities"] = tuple(values["probabilities"])
    values["thresholds"] = tuple(values["thresholds"])
    values["bins"] = bins
    values["blocks"] = tuple(values["blocks"])
    values["merge_events"] = tuple(values["merge_events"])
    values.setdefault("score_semantics", {})
    try:
        artifact = CalibrationArtifact(**values)
    except TypeError as exc:
        raise ArtifactError(f"invalid artifact shape: {exc}") from exc
    if any(left > right for left, right in zip(artifact.probabilities, artifact.probabilities[1:], strict=False)):
        raise ArtifactError("stored artifact probabilities are not non-decreasing")
    return artifact


def artifact_from_json(payload: str) -> CalibrationArtifact:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ArtifactError("artifact payload is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ArtifactError("artifact payload must be an object")
    return artifact_from_dict(data)
