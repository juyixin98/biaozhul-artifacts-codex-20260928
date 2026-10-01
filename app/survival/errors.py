"""Domain errors and structured notices.

The API maps these codes to stable failure categories while statistical
non-failures are returned separately as warnings or uncertain conclusions.
"""

from __future__ import annotations


class SurvivalValidationError(ValueError):
    code = "invalid_survival_input"


class EmptyDatasetError(SurvivalValidationError):
    code = "empty_dataset"


class UnknownGroupError(SurvivalValidationError):
    code = "unknown_group"


class DuplicateGroupError(SurvivalValidationError):
    code = "duplicate_group"


class InsufficientGroupsError(SurvivalValidationError):
    code = "insufficient_groups"


class EmptyGroupError(SurvivalValidationError):
    code = "empty_declared_group"
