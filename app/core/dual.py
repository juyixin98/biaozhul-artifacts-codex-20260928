from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Dual:
    """A scalar dual number: real + dot(grad, parameter_increment)."""

    real: float
    grad: np.ndarray


def _is_dual(value: object) -> bool:
    return isinstance(value, Dual)


def _real(value: Dual | float) -> float:
    return value.real if isinstance(value, Dual) else float(value)


def _grad(value: Dual | float) -> np.ndarray:
    return value.grad if isinstance(value, Dual) else np.zeros(0)


def _gradient_pair(
    a: Dual | float,
    b: Dual | float,
) -> tuple[np.ndarray, np.ndarray]:
    ga = _grad(a)
    gb = _grad(b)
    if ga.size == 0 and gb.size:
        ga = np.zeros_like(gb)
    if gb.size == 0 and ga.size:
        gb = np.zeros_like(ga)
    return ga, gb


def add(a: Dual | float, b: Dual | float) -> Dual | float:
    if not (_is_dual(a) or _is_dual(b)):
        return _real(a) + _real(b)
    ga, gb = _gradient_pair(a, b)
    return Dual(_real(a) + _real(b), ga + gb)


def sub(a: Dual | float, b: Dual | float) -> Dual | float:
    if not (_is_dual(a) or _is_dual(b)):
        return _real(a) - _real(b)
    ga, gb = _gradient_pair(a, b)
    return Dual(_real(a) - _real(b), ga - gb)


def mul(a: Dual | float, b: Dual | float) -> Dual | float:
    if not (_is_dual(a) or _is_dual(b)):
        return _real(a) * _real(b)
    ar, br = _real(a), _real(b)
    ga, gb = _gradient_pair(a, b)
    return Dual(ar * br, br * ga + ar * gb)


def div(a: Dual | float, b: Dual | float) -> Dual | float:
    br = _real(b)
    if br == 0.0:
        raise ZeroDivisionError("division by zero in model expression")
    if not (_is_dual(a) or _is_dual(b)):
        return _real(a) / br
    ga, gb = _gradient_pair(a, b)
    return Dual(
        _real(a) / br,
        (ga * br - _real(a) * gb) / (br * br),
    )


def neg(a: Dual | float) -> Dual | float:
    return Dual(-a.real, -a.grad) if isinstance(a, Dual) else -float(a)


def pos(a: Dual | float) -> Dual | float:
    return a


def power(a: Dual | float, b: Dual | float) -> Dual | float:
    ar, br = _real(a), _real(b)
    if isinstance(b, Dual):
        if ar <= 0.0:
            raise ValueError("a variable exponent requires a strictly positive base")
        real = ar**br
        log_a = math.log(ar)
        ga, gb = _gradient_pair(a, b)
        grad = real * (gb * log_a + (b.real / ar) * ga)
        return Dual(real, grad)

    exponent = float(b)
    integer_exponent = float(exponent).is_integer()
    if not integer_exponent and ar < 0.0:
        raise ValueError("fractional power of a negative value is not real-valued")
    if ar == 0.0 and exponent < 1.0:
        raise ValueError("power is not differentiable at zero for this exponent")
    real = ar**exponent
    if not _is_dual(a):
        return real
    if ar == 0.0 and exponent == 1.0:
        derivative = 1.0
    else:
        derivative = exponent * ar ** (exponent - 1.0)
    return Dual(real, derivative * a.grad)


def _unary_function(
    value: Dual | float,
    real_fn,
    derivative_fn,
    name: str,
) -> Dual | float:
    real_value = _real(value)
    real = real_fn(real_value)
    if not _is_dual(value):
        return real
    derivative = derivative_fn(real_value)
    return Dual(real, derivative * value.grad)


FUNCTIONS = {
    "exp": (math.exp, lambda x: math.exp(x)),
    "log": (math.log, lambda x: 1.0 / x),
    "sqrt": (math.sqrt, lambda x: 0.5 / math.sqrt(x)),
    "sin": (math.sin, math.cos),
    "cos": (math.cos, lambda x: -math.sin(x)),
    "tan": (math.tan, lambda x: 1.0 / math.cos(x) ** 2),
    "tanh": (math.tanh, lambda x: 1.0 - math.tanh(x) ** 2),
    "abs": (abs, lambda x: 1.0 if x > 0 else -1.0),
}


def call_function(name: str, value: Dual | float) -> Dual | float:
    if name not in FUNCTIONS:
        raise ValueError(f"unsupported function: {name}")
    real_fn, derivative_fn = FUNCTIONS[name]
    try:
        return _unary_function(value, real_fn, derivative_fn, name)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"invalid value in {name}: {_real(value)}") from exc
