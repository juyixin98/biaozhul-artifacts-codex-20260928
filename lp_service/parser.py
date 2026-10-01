"""Convert API models into a dense, validated original LP."""

from dataclasses import dataclass

import numpy as np

from .models import LPRequest


@dataclass(frozen=True)
class LPProblem:
    c: np.ndarray
    sense: str
    A: np.ndarray
    constraint_senses: np.ndarray
    b: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    variable_names: tuple[str, ...]
    constraint_names: tuple[str | None, ...]


def parse_request(request: LPRequest) -> LPProblem:
    n = len(request.objective)
    variable_names = tuple(f"x{j}" for j in range(n))
    c = np.asarray(request.objective, dtype=float)

    rows = [_dense_row(row.coefficients, variable_names, n) for row in request.constraints]
    A = np.vstack(rows) if rows else np.empty((0, n), dtype=float)
    constraint_senses = np.asarray([row.sense for row in request.constraints], dtype=object)
    b = np.asarray([row.rhs for row in request.constraints], dtype=float)
    constraint_names = tuple(row.name for row in request.constraints)

    lower = np.full(n, -np.inf, dtype=float)
    upper = np.full(n, np.inf, dtype=float)
    for j, bound in enumerate(request.bounds):
        if bound is None:
            continue
        if bound.lower is not None:
            lower[j] = bound.lower
        if bound.upper is not None:
            upper[j] = bound.upper

    return LPProblem(
        c=c,
        sense=request.sense,
        A=A,
        constraint_senses=constraint_senses,
        b=b,
        lower=lower,
        upper=upper,
        variable_names=variable_names,
        constraint_names=constraint_names,
    )


def _dense_row(
    coefficients: dict[str, float] | list[float],
    variable_names: tuple[str, ...],
    n: int,
) -> np.ndarray:
    if isinstance(coefficients, list):
        return np.asarray(coefficients, dtype=float)

    row = np.zeros(n, dtype=float)
    for name, value in coefficients.items():
        try:
            index = variable_names.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown variable {name!r}") from exc
        row[index] = value
    return row
