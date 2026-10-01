"""Parsing and exact numeric policy.

JSON integers and decimal strings become exact fractions.  JSON floating-point
values are interpreted as their exact binary value rather than silently applying
a comparison epsilon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from decimal import ROUND_HALF_EVEN
from fractions import Fraction
from typing import Any, Literal

from app.errors import ValidationFailure

Direction = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class NumericPolicy:
    mode: Literal["exact", "quantized"]
    tolerance: Fraction | None = None

    @property
    def is_exact(self) -> bool:
        return self.mode == "exact"

    def normalize_transformed(self, value: Fraction) -> Fraction:
        """Normalize a direction-adjusted value.

        Quantized mode rounds to an exact multiple of the tolerance and then
        compares with ordinary strict order.  Consequently dominance remains
        transitive; the approximation is reported separately as uncertainty.
        """
        if self.mode == "exact" or self.tolerance is None:
            return value
        quotient = Decimal(value.numerator) / Decimal(value.denominator)
        step = Decimal(self.tolerance.numerator) / Decimal(self.tolerance.denominator)
        snapped = (quotient / step).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN) * step
        return Fraction(snapped)


EXACT_POLICY = NumericPolicy(mode="exact")


def direction_sign(direction: Direction) -> int:
    if direction == "min":
        return -1
    if direction == "max":
        return 1
    raise ValidationFailure(
        f"Unsupported optimization direction {direction!r}.",
        code="invalid_direction",
        details={"allowed": ["min", "max"], "received": direction},
    )


def parse_fraction(value: Any, *, location: str) -> Fraction:
    if isinstance(value, bool):
        raise ValidationFailure(
            "Boolean values are not valid objective values.",
            code="invalid_objective_value",
            details={"location": location, "received": value},
        )
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationFailure(
                "Objective values must be finite.",
                code="non_finite_objective",
                details={"location": location},
            )
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            decimal_value = Decimal(text)
        except Exception:
            decimal_value = None
        if decimal_value is None or not decimal_value.is_finite():
            raise ValidationFailure(
                "Objective string must be a finite decimal number.",
                code="invalid_objective_value",
                details={"location": location, "received": value},
            )
        return Fraction(decimal_value)
    raise ValidationFailure(
        "Objective value must be an integer, finite float, or decimal string.",
        code="invalid_objective_value",
        details={"location": location, "type": type(value).__name__},
    )


def build_policy(
    mode: Literal["exact", "quantized"] = "exact",
    tolerance: Any = None,
) -> NumericPolicy:
    if mode == "exact":
        if tolerance is not None:
            raise ValidationFailure(
                "Set numeric mode 'quantized' to supply a comparison tolerance; "
                "exact mode does not silently apply an epsilon.",
                code="tolerance_requires_quantized_mode",
                details={"tolerance": tolerance},
            )
        return EXACT_POLICY

    if tolerance is None:
        raise ValidationFailure(
            "Quantized numeric mode requires a positive tolerance.",
            code="missing_tolerance",
        )
    tolerance_value = parse_fraction(tolerance, location="numeric_settings.tolerance")
    if tolerance_value <= 0:
        raise ValidationFailure(
            "Comparison tolerance must be greater than zero.",
            code="invalid_tolerance",
            details={"received": str(tolerance_value)},
        )
    return NumericPolicy(mode="quantized", tolerance=tolerance_value)


def fraction_to_string(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
