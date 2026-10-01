"""Shared validation rules and numeric limits."""

from __future__ import annotations

import math

from app.errors import (
    InvalidInputError,
    PayloadTooLargeError,
    ResourceExhaustedError,
)

# SQLite integers are signed 64-bit values. Larger batch counts cannot be stored
# without changing the persistence contract.
MAX_BATCH_COUNT = 2**63 - 1

# A prediction allocates one float64 for each k=0..n. Ten thousand values are
# enough for local fixtures while keeping memory use bounded.
MAX_PREDICTION_N = 10_000


def strict_int(value: object, field_name: str) -> int:
    """Accept only genuine integers; booleans and strings are sample errors."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(
            f"{field_name} must be an integer.",
            {"field": field_name, "received_type": type(value).__name__},
        )
    return value


def ensure_count(value: object, field_name: str = "count") -> int:
    value = strict_int(value, field_name)
    if value < 0:
        raise InvalidInputError(
            f"{field_name} must be non-negative.",
            {"field": field_name, "received": value},
        )
    if value > MAX_BATCH_COUNT:
        raise ResourceExhaustedError(
            f"{field_name} exceeds the signed 64-bit SQLite limit.",
            {"field": field_name, "received": value, "limit": MAX_BATCH_COUNT},
        )
    return value


def ensure_future_n(value: object) -> int:
    value = ensure_count(value, "future_n")
    if value > MAX_PREDICTION_N:
        raise PayloadTooLargeError(
            "future_n would allocate too many prediction points.",
            {"future_n": value, "max_future_n": MAX_PREDICTION_N},
        )
    return value


def ensure_positive_param(value: object, field_name: str) -> float:
    """Validate a Beta prior parameter.

    Integers are accepted because JSON commonly encodes one as 1. Booleans and
    strings are not. Parameters must be finite and strictly positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            f"{field_name} must be a positive number.",
            {"field": field_name, "received_type": type(value).__name__},
        )
    try:
        number = float(value)
    except OverflowError as exc:
        raise ResourceExhaustedError(
            f"{field_name} is too large to represent as float64.",
            {"field": field_name},
        ) from exc
    if not math.isfinite(number):
        raise InvalidInputError(
            f"{field_name} must be finite.",
            {"field": field_name, "received": str(value)},
        )
    if number <= 0.0:
        raise InvalidInputError(
            f"{field_name} must be strictly positive.",
            {"field": field_name, "received": number},
        )
    return number


def ensure_prior(alpha: object, beta: object) -> tuple[float, float]:
    return (
        ensure_positive_param(alpha, "alpha"),
        ensure_positive_param(beta, "beta"),
    )


def ensure_level(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            "confidence_level must be a number.",
            {"received_type": type(value).__name__},
        )
    level = float(value)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise InvalidInputError(
            "confidence_level must be strictly between 0 and 1.",
            {"received": level},
        )
    return level
