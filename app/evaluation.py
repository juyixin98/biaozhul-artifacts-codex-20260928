from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .errors import ArtifactError, DatasetLeakageError
from .prediction import Prediction
from .validation import CleanPoint

EPSILON = 1e-15


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    count: int
    total_weight: float
    weighted_brier_score: float
    weighted_log_loss: float
    weighted_absolute_error: float
    extrapolated_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_weight": self.total_weight,
            "weighted_brier_score": self.weighted_brier_score,
            "weighted_log_loss": self.weighted_log_loss,
            "weighted_absolute_error": self.weighted_absolute_error,
            "extrapolated_count": self.extrapolated_count,
        }


def dataset_fingerprint(points: list[CleanPoint]) -> str:
    payload = [
        (point.score, point.target, point.weight, point.record_id)
        for point in sorted(points, key=lambda item: (item.score, item.target, item.weight, item.record_id or ""))
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_separate_dataset(*, artifact_fingerprint: str, evaluation_points: list[CleanPoint]) -> None:
    evaluation_fingerprint = dataset_fingerprint(evaluation_points)
    if evaluation_fingerprint == artifact_fingerprint:
        raise DatasetLeakageError(
            "evaluation set must be separate from calibration set",
            details={"fingerprint": evaluation_fingerprint},
        )


def evaluate_predictions(
    points: list[CleanPoint],
    predictions: list[Prediction],
    *,
    artifact: CalibrationArtifact,
    target_name: str,
) -> EvaluationMetrics:
    if target_name != artifact.target_name:
        raise ArtifactError(
            "target_name does not match trained artifact",
            details={"requested_target_name": target_name, "artifact_target_name": artifact.target_name},
        )
    if len(points) != len(predictions):
        raise ArtifactError("evaluation points and predictions must have the same length")

    total_weight = sum(point.weight for point in points)
    brier = 0.0
    log_loss = 0.0
    absolute_error = 0.0
    extrapolated_count = 0
    for point, prediction in zip(points, predictions, strict=True):
        if point.record_id != prediction.record_id or point.score != prediction.score:
            raise ArtifactError("prediction order does not match evaluation records")
        probability = min(max(prediction.probability, EPSILON), 1.0 - EPSILON)
        error = prediction.probability - point.target
        brier += point.weight * error * error
        log_loss += point.weight * -(
            point.target * math.log(probability) + (1.0 - point.target) * math.log(1.0 - probability)
        )
        absolute_error += point.weight * abs(error)
        extrapolated_count += int(prediction.extrapolated)

    return EvaluationMetrics(
        count=len(points),
        total_weight=total_weight,
        weighted_brier_score=brier / total_weight,
        weighted_log_loss=log_loss / total_weight,
        weighted_absolute_error=absolute_error / total_weight,
        extrapolated_count=extrapolated_count,
    )
