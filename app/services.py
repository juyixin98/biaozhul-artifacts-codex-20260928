from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from .artifacts import CalibrationArtifact, artifact_from_json, build_artifact
from .config import Settings
from .evaluation import dataset_fingerprint, ensure_separate_dataset, evaluate_predictions
from .kernel import fit_pav
from .prediction import predict_scores
from .repository import SqliteModelRepository
from .validation import validate_records


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_model_id() -> str:
    return uuid.uuid4().hex[:12]


class CalibrationService:
    def __init__(self, repository: SqliteModelRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings

    def train(self, request: dict[str, Any]) -> tuple[CalibrationArtifact, list[dict[str, Any]]]:
        raw_records = list(request["records"])
        points, warning_issues = validate_records(raw_records, require_target=True)
        zero_weight_count = len(raw_records) - len(points)
        pav = fit_pav(points)
        artifact = build_artifact(
            model_id=new_model_id(),
            created_at_utc=utc_now(),
            score_name=request["score_name"],
            target_name=request["target_name"],
            extrapolation=request.get("extrapolation", "clip"),
            pav=pav,
            input_record_count=len(raw_records),
            zero_weight_count=zero_weight_count,
            training_fingerprint=dataset_fingerprint(points),
            schema_version=self.settings.artifact_schema_version,
        )
        self.repository.save_model(artifact)
        warnings = [issue.as_dict() for issue in warning_issues]
        return artifact, warnings

    def load_artifact(self, model_id: str) -> CalibrationArtifact:
        payload = self.repository.get_artifact_json(model_id)
        if payload is None:
            from .errors import ModelNotFound

            raise ModelNotFound("model_id does not exist", details={"model_id": model_id})
        return artifact_from_json(payload)

    def predict(
        self, request: dict[str, Any]
    ) -> tuple[CalibrationArtifact, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        artifact = self.load_artifact(request["model_id"])
        # Prediction input has no target; validation still rejects malformed records.
        prediction_records = []
        for record in request["records"]:
            clean_record = {"score": record["score"], "weight": 1.0}
            if record.get("record_id") is not None:
                clean_record["record_id"] = record["record_id"]
            prediction_records.append(clean_record)
        points, warnings = validate_records(prediction_records, require_target=False)
        score_tuples = [(index, point.score, point.record_id) for index, point in enumerate(points)]
        predictions, uncertainties = predict_scores(
            artifact,
            score_tuples,
            score_name=request["score_name"],
            mode=request.get("extrapolation"),
        )
        return (
            artifact,
            [self._prediction_dict(item) for item in predictions],
            uncertainties,
            [issue.as_dict() for issue in warnings],
        )

    def evaluate(self, request: dict[str, Any]) -> tuple[CalibrationArtifact, list[dict[str, Any]], dict[str, Any]]:
        artifact = self.load_artifact(request["model_id"])
        points, _ = validate_records(request["records"], require_target=True)
        ensure_separate_dataset(artifact_fingerprint=artifact.training_fingerprint, evaluation_points=points)
        predictions, _ = predict_scores(
            artifact,
            [(index, point.score, point.record_id) for index, point in enumerate(points)],
            score_name=request["score_name"],
            mode="clip",
        )
        metrics = evaluate_predictions(
            points,
            predictions,
            artifact=artifact,
            target_name=request["target_name"],
        )
        return artifact, [self._prediction_dict(item) for item in predictions], metrics.as_dict()

    @staticmethod
    def _prediction_dict(prediction: Any) -> dict[str, Any]:
        return {
            "index": prediction.index,
            "record_id": prediction.record_id,
            "score": prediction.score,
            "probability": prediction.probability,
            "bin_index": prediction.bin_index,
            "extrapolated": prediction.extrapolated,
        }
