from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import CalibrationArtifact
from .errors import ArtifactError, ExtrapolationRejected


@dataclass(frozen=True, slots=True)
class Prediction:
    index: int
    score: float
    probability: float
    bin_index: int
    extrapolated: bool
    record_id: str | None = None


def predict_scores(
    artifact: CalibrationArtifact,
    scores: list[tuple[int, float, str | None]],
    *,
    score_name: str,
    mode: str | None = None,
) -> tuple[list[Prediction], list[dict[str, object]]]:
    if score_name != artifact.score_name:
        raise ArtifactError(
            "score_name does not match trained artifact",
            details={"requested_score_name": score_name, "artifact_score_name": artifact.score_name},
        )
    effective_mode = mode or artifact.extrapolation
    if effective_mode not in {"clip", "raise"}:
        raise ArtifactError("extrapolation must be 'clip' or 'raise'")

    thresholds = np.array(artifact.thresholds, dtype=float)
    probabilities = np.array(artifact.probabilities, dtype=float)
    predictions: list[Prediction] = []
    uncertainties: list[dict[str, object]] = []

    for index, score, record_id in scores:
        below = score < artifact.score_min
        above = score > artifact.score_max
        outside = below or above
        if outside and effective_mode == "raise":
            raise ExtrapolationRejected(
                "score is outside the calibrated score range and extrapolation is disabled",
                details={
                    "index": index,
                    "record_id": record_id,
                    "score": score,
                    "score_min": artifact.score_min,
                    "score_max": artifact.score_max,
                },
            )

        if below:
            bin_index = 0
        elif above:
            bin_index = len(probabilities) - 1
        else:
            # score == upper_threshold belongs to the lower piece.
            bin_index = int(np.searchsorted(thresholds, score, side="right"))
        predictions.append(
            Prediction(
                index=index,
                score=score,
                probability=float(probabilities[bin_index]),
                bin_index=bin_index,
                extrapolated=outside,
                record_id=record_id,
            )
        )
        if outside:
            uncertainties.append(
                {
                    "code": "extrapolated_probability",
                    "message": "probability is held at the nearest calibrated endpoint; this is not observed in training",
                    "index": index,
                    "record_id": record_id,
                    "policy": "clip",
                    "score_min": artifact.score_min,
                    "score_max": artifact.score_max,
                }
            )

    return predictions, uncertainties
