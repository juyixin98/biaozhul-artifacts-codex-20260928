"""Domain exceptions with stable failure categories for API and tests."""

from __future__ import annotations


class SurveyImputationError(ValueError):
    category = "invalid_request"


class EmptyDatasetError(SurveyImputationError):
    category = "empty_dataset"


class MissingOutcomeError(SurveyImputationError):
    category = "missing_outcome_column"


class MaskObservedMismatchError(SurveyImputationError):
    category = "mask_observed_mismatch"


class NotMissingMaskedError(SurveyImputationError):
    category = "not_missing_but_masked"


class ObservedMissingInMaskError(SurveyImputationError):
    category = "observed_value_missing_in_mask"


class MaskShapeError(SurveyImputationError):
    category = "mask_shape_mismatch"


class AllOutcomeMissingError(SurveyImputationError):
    category = "all_outcome_missing"


class InsufficientObservedRowsError(SurveyImputationError):
    category = "insufficient_observed_rows"


class InvalidParameterError(SurveyImputationError):
    category = "invalid_parameter"


class ResultNotFoundError(SurveyImputationError):
    category = "result_not_found"


class ResultAlreadyExistsError(SurveyImputationError):
    category = "result_already_exists"
