"""Domain and API error categories."""

from __future__ import annotations


class AnalysisError(ValueError):
    """Base class for expected analysis failures."""

    code = "ANALYSIS_ERROR"
    http_status = 400


class InvalidSeriesError(AnalysisError):
    code = "INVALID_SERIES"


class InvalidLagGridError(AnalysisError):
    code = "INVALID_LAG_GRID"


class DuplicateRequestError(AnalysisError):
    code = "DUPLICATE_REQUEST_ID"
    http_status = 409


class ResultNotFoundError(AnalysisError):
    code = "RESULT_NOT_FOUND"
    http_status = 404
