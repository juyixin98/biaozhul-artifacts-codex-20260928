from __future__ import annotations


class AnalysisError(ValueError):
    """Base class for deterministic analysis validation failures."""

    error_code = "analysis_error"
    status_code = 422


class EmptyDatasetError(AnalysisError):
    error_code = "empty_dataset"


class DuplicateRecordIdError(AnalysisError):
    error_code = "duplicate_record_id"


class IneligibleObservationError(AnalysisError):
    error_code = "no_eligible_observations"


class AllMissingFieldError(AnalysisError):
    error_code = "all_missing_field"


class MissingOutcomeError(AnalysisError):
    error_code = "missing_outcome"


class InsufficientRowsError(AnalysisError):
    error_code = "insufficient_rows"


class MissingAllPredictorsError(AnalysisError):
    error_code = "missing_all_predictors"


class SingularDesignError(AnalysisError):
    error_code = "singular_design"


class InvalidParameterError(AnalysisError):
    error_code = "invalid_parameter"
