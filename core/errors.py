"""Stable error categories returned by validation and statistical services."""

from __future__ import annotations


class CalibrationError(ValueError):
    """Base class for expected calibration errors."""

    category = "calibration_error"


class SampleContractError(CalibrationError):
    category = "sample_contract_error"


class ProbabilityRangeError(SampleContractError):
    category = "probability_range_error"


class WeightError(SampleContractError):
    category = "weight_error"


class BinningError(SampleContractError):
    category = "binning_error"


class EmptyCalibrationBin(CalibrationError):
    category = "empty_calibration_bin"


class RunNotFoundError(LookupError):
    category = "run_not_found"
