from __future__ import annotations

import pytest

from app.errors import ValidationFailure
from app.parsing import build_policy, parse_fraction


def test_exact_policy_rejects_tolerance_instead_of_silent_epsilon():
    with pytest.raises(ValidationFailure) as error:
        build_policy("exact", "0.001")
    assert error.value.code == "tolerance_requires_quantized_mode"


def test_quantized_policy_requires_positive_tolerance():
    with pytest.raises(ValidationFailure) as zero:
        build_policy("quantized", "0")
    assert zero.value.code == "invalid_tolerance"

    with pytest.raises(ValidationFailure) as missing:
        build_policy("quantized")
    assert missing.value.code == "missing_tolerance"


def test_boolean_and_non_finite_values_are_rejected():
    with pytest.raises(ValidationFailure) as bool_error:
        parse_fraction(True, location="test")
    assert bool_error.value.code == "invalid_objective_value"

    with pytest.raises(ValidationFailure) as nan_error:
        parse_fraction(float("nan"), location="test")
    assert nan_error.value.code == "non_finite_objective"


def test_decimal_string_is_exact_half_even_grid_quantization():
    policy = build_policy("quantized", "1")
    assert policy.normalize_transformed(parse_fraction("0.5", location="x")) == 0
    assert policy.normalize_transformed(parse_fraction("1.5", location="x")) == 2
    assert policy.normalize_transformed(parse_fraction("2.5", location="x")) == 2
