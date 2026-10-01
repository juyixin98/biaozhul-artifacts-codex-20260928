class ChangeDetectionError(Exception):
    """Base class for expected domain failures."""

    category = "UNKNOWN"
    status_code = 500

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidSeriesError(ChangeDetectionError):
    category = "INVALID_SERIES"
    status_code = 422


class InvalidParameterError(ChangeDetectionError):
    category = "INVALID_PARAMETER"
    status_code = 422


class NumericStabilityError(ChangeDetectionError):
    category = "NUMERIC_STABILITY"
    status_code = 500


class PruningVerificationError(ChangeDetectionError):
    category = "PRUNING_MISMATCH"
    status_code = 500


class PersistenceError(ChangeDetectionError):
    category = "PERSISTENCE_ERROR"
    status_code = 500


class RunNotFoundError(ChangeDetectionError):
    category = "RUN_NOT_FOUND"
    status_code = 404

