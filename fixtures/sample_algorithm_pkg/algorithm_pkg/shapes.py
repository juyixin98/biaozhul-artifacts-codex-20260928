"""Conditional behavior fixture."""


def classify_size(value: int) -> str:
    if value >= 10:
        return "large"
    if value > 0:
        return "small"
    return "zero-or-negative"
