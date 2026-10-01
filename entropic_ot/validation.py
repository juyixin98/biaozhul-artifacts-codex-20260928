"""Input parsing and validation contracts for finite discrete OT."""

from dataclasses import dataclass

import numpy as np

from .errors import InputValidationError


@dataclass(frozen=True, slots=True)
class ValidatedProblem:
    source_mass: np.ndarray
    target_mass: np.ndarray
    cost: np.ndarray
    epsilon: float
    max_iterations: int
    total_mass: float
    active_rows: np.ndarray
    active_columns: np.ndarray

    @property
    def active_cost(self) -> np.ndarray:
        return self.cost[np.ix_(self.active_rows, self.active_columns)]


def _as_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise InputValidationError(
            f"{name} must be a one-dimensional array.",
            details={"field": name, "actual_ndim": array.ndim},
        )
    if array.size == 0:
        raise InputValidationError(
            f"{name} must contain at least one entry.",
            details={"field": name},
        )
    return array


def validate_problem(
    source_mass,
    target_mass,
    cost,
    *,
    epsilon: float,
    max_iterations: int,
    mass_atol: float = 1e-12,
    mass_rtol: float = 1e-9,
) -> ValidatedProblem:
    """Validate masses, shape, finiteness, regularization, and mass balance."""

    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "epsilon must be a finite positive number.",
            details={"field": "epsilon"},
        ) from exc

    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise InputValidationError(
            "epsilon must be a finite positive number.",
            details={"field": "epsilon", "actual": epsilon_value},
        )

    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise InputValidationError(
            "max_iterations must be an integer.",
            details={"field": "max_iterations", "actual_type": type(max_iterations).__name__},
        )
    if max_iterations <= 0:
        raise InputValidationError(
            "max_iterations must be positive.",
            details={"field": "max_iterations", "actual": max_iterations},
        )

    for tolerance_name, tolerance in (
        ("mass_atol", mass_atol),
        ("mass_rtol", mass_rtol),
    ):
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
            raise InputValidationError(
                f"{tolerance_name} must be a number.",
                details={"field": tolerance_name},
            )
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise InputValidationError(
                f"{tolerance_name} must be finite and non-negative.",
                details={"field": tolerance_name, "actual": float(tolerance)},
            )

    a = _as_vector(source_mass, "source_mass")
    b = _as_vector(target_mass, "target_mass")
    c = np.asarray(cost, dtype=np.float64)

    if c.ndim != 2:
        raise InputValidationError(
            "cost must be a two-dimensional matrix.",
            details={"field": "cost", "actual_ndim": c.ndim},
        )
    if c.shape != (a.size, b.size):
        raise InputValidationError(
            "cost shape must match the source and target masses.",
            details={
                "field": "cost",
                "cost_shape": list(c.shape),
                "expected_shape": [a.size, b.size],
            },
        )
    if not np.all(np.isfinite(a)):
        raise InputValidationError(
            "source_mass contains NaN or infinite values.",
            details={"field": "source_mass"},
        )
    if not np.all(np.isfinite(b)):
        raise InputValidationError(
            "target_mass contains NaN or infinite values.",
            details={"field": "target_mass"},
        )
    if not np.all(np.isfinite(c)):
        raise InputValidationError(
            "cost contains NaN or infinite values.",
            details={"field": "cost"},
        )
    if np.any(a < 0.0):
        raise InputValidationError(
            "source_mass must be non-negative.",
            details={"field": "source_mass"},
        )
    if np.any(b < 0.0):
        raise InputValidationError(
            "target_mass must be non-negative.",
            details={"field": "target_mass"},
        )

    source_total = float(np.sum(a))
    target_total = float(np.sum(b))
    total_scale = max(abs(source_total), abs(target_total), 1.0)
    mass_tolerance = float(mass_atol + mass_rtol * total_scale)
    if abs(source_total - target_total) > mass_tolerance:
        raise InputValidationError(
            "source and target total masses must match.",
            details={
                "field": "mass_balance",
                "source_total": source_total,
                "target_total": target_total,
                "tolerance": mass_tolerance,
            },
        )

    active_rows = a > 0.0
    active_columns = b > 0.0

    return ValidatedProblem(
        source_mass=a,
        target_mass=b,
        cost=c,
        epsilon=epsilon_value,
        max_iterations=int(max_iterations),
        total_mass=source_total,
        active_rows=active_rows,
        active_columns=active_columns,
    )
