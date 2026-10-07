from __future__ import annotations


class CalibrationError(Exception):
    """Base class for expected calibration failures."""

    error_code = "calibration_error"
    status_code = 400

    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ValidationFailure(CalibrationError):
    error_code = "validation_failed"
    status_code = 422


class ArtifactError(CalibrationError):
    error_code = "artifact_error"
    status_code = 400


class ModelNotFound(CalibrationError):
    error_code = "model_not_found"
    status_code = 404


class ExtrapolationRejected(CalibrationError):
    error_code = "extrapolation_rejected"
    status_code = 422


class DatasetLeakageError(CalibrationError):
    error_code = "evaluation_dataset_overlap"
    status_code = 422
