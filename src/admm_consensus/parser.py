"""Parse and validate separable quadratic models from API payloads."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import Settings
from .errors import InputError, ResourceExhausted
from .models import (
    AdaptiveRho,
    Problem,
    QuadraticModel,
    SolverConfig,
    Tolerances,
    positive_finite,
)


def _vector(name: str, value: Any, dimension: int | None = None) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a numeric vector.") from exc
    if arr.ndim != 1:
        raise InputError(f"{name} must be one-dimensional.", {"shape": list(arr.shape)})
    if dimension is not None and arr.size != dimension:
        raise InputError(
            f"{name} has dimension {arr.size}, expected {dimension}.",
            {"shape": list(arr.shape)},
        )
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} contains NaN or infinity.")
    return arr


def _matrix(name: str, value: Any, dimension: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a numeric matrix.") from exc
    if arr.ndim != 2 or arr.shape != (dimension, dimension):
        raise InputError(
            f"{name} must be a {dimension}x{dimension} matrix.",
            {"shape": list(arr.shape)},
        )
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} contains NaN or infinity.")
    return arr


def _positive_definite(Q: np.ndarray, label: str) -> np.ndarray:
    symmetric = 0.5 * (Q + Q.T)
    asymmetry = float(np.max(np.abs(Q - Q.T))) if Q.size else 0.0
    if asymmetry > 1e-10:
        raise InputError(f"{label}.Q must be symmetric.", {"max_abs_asymmetry": asymmetry})
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise InputError(f"{label}.Q cannot be inspected numerically.") from exc
    eig_min = float(eigenvalues[0])
    eig_max = float(eigenvalues[-1])
    required_min = max(1e-10, 1e-9 * max(1.0, abs(eig_max)))
    if eig_min <= required_min:
        raise InputError(
            f"{label}.Q must be positive definite.",
            {"min_eigenvalue": eig_min, "required_min_eigenvalue": required_min},
        )
    return symmetric


def parse_problem(payload: dict[str, Any], settings: Settings) -> Problem:
    workers = payload.get("workers")
    if not isinstance(workers, list):
        raise InputError("workers must be a list.")
    if len(workers) != 2:
        raise InputError("This teaching backend requires exactly two local compute processes.", {"workers": len(workers)})

    parsed: list[QuadraticModel] = []
    expected_dimension: int | None = payload.get("dimension")
    if expected_dimension is not None:
        if not isinstance(expected_dimension, int) or expected_dimension <= 0:
            raise InputError("dimension must be a positive integer.")
        if expected_dimension > settings.max_dimension:
            raise ResourceExhausted(
                "Requested dimension exceeds the local teaching limit.",
                {"dimension": expected_dimension, "limit": settings.max_dimension},
            )

    for index, worker in enumerate(workers, start=1):
        if not isinstance(worker, dict):
            raise InputError(f"workers[{index}] must be an object.")
        worker_id = worker.get("worker_id", index)
        if worker_id != index:
            raise InputError(
                "Worker IDs must be 1 and 2 in stable process order.",
                {"index": index, "worker_id": worker_id},
            )
        q = _vector(f"workers[{index}].q", worker.get("q"), expected_dimension)
        dimension = q.size
        if expected_dimension is None:
            if dimension > settings.max_dimension:
                raise ResourceExhausted(
                    "Requested dimension exceeds the local teaching limit.",
                    {"dimension": dimension, "limit": settings.max_dimension},
                )
            expected_dimension = dimension
        Q = _matrix(f"workers[{index}].Q", worker.get("Q"), expected_dimension)
        Q = _positive_definite(Q, f"workers[{index}]")
        parsed.append(QuadraticModel(Q=Q, q=q))

    if not parsed:
        raise InputError("At least one worker model is required.")
    return Problem(models=(parsed[0], parsed[1]), dimension=expected_dimension or 0)


def parse_config(payload: dict[str, Any], settings: Settings) -> SolverConfig:
    try:
        max_iterations = int(payload.get("max_iterations", 100))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError("max_iterations must be a positive integer.") from exc
    if max_iterations <= 0:
        raise InputError("max_iterations must be positive.")
    if max_iterations > settings.max_iterations:
        raise ResourceExhausted(
            "Requested iteration count exceeds the local teaching limit.",
            {"max_iterations": max_iterations, "limit": settings.max_iterations},
        )

    rho = positive_finite(float(payload.get("rho", 1.0)), "rho")
    tolerance_payload = payload.get("tolerances", {})
    if tolerance_payload is None:
        tolerance_payload = {}
    if not isinstance(tolerance_payload, dict):
        raise InputError("tolerances must be an object.")
    primal_abs = positive_finite(float(tolerance_payload.get("primal_abs", 1e-7)), "tolerances.primal_abs")
    dual_abs = positive_finite(float(tolerance_payload.get("dual_abs", 1e-7)), "tolerances.dual_abs")
    primal_rel = float(tolerance_payload.get("primal_rel", 0.0))
    dual_rel = float(tolerance_payload.get("dual_rel", 0.0))
    if primal_rel < 0.0 or dual_rel < 0.0 or not np.isfinite(primal_rel) or not np.isfinite(dual_rel):
        raise InputError("Relative tolerances must be finite and non-negative.")

    adaptive = AdaptiveRho.from_api(payload.get("adaptive_rho"))
    if adaptive is not None:
        adaptive.validate()
        if any(round_number > max_iterations for round_number in adaptive.change_rounds):
            raise InputError("adaptive rho change round cannot exceed max_iterations.")

    return SolverConfig(
        max_iterations=max_iterations,
        initial_rho=rho,
        tol=Tolerances(
            primal_abs=primal_abs,
            dual_abs=dual_abs,
            primal_rel=primal_rel,
            dual_rel=dual_rel,
        ),
        adaptive_rho=adaptive,
    )
