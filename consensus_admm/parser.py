"""Parsing and validation for separable convex quadratic objectives.

Each worker contributes
    f_i(x) = 0.5 * x.T @ P_i @ x + q_i.T @ x + r_i,
with P_i symmetric positive semidefinite and all local variables sharing one
consensus dimension.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errors import ModelParseError
from .models import QuadraticModel, QuadraticWorker


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelParseError(
            f"{field} must be a finite number", details={"field": field}
        ) from exc
    if not np.isfinite(result):
        raise ModelParseError(
            f"{field} must be finite", details={"field": field}
        )
    return result


def parse_vector(value: Any, field: str, dimension: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelParseError(f"{field} must be a numeric vector") from exc
    if result.ndim != 1 or result.shape[0] != dimension:
        raise ModelParseError(
            f"{field} must have length {dimension}",
            details={"field": field, "actual_shape": list(result.shape)},
        )
    if not np.all(np.isfinite(result)):
        raise ModelParseError(f"{field} contains a non-finite value")
    return result


def parse_matrix(value: Any, field: str, dimension: int) -> np.ndarray:
    if isinstance(value, Mapping):
        kind = value.get("type", "full")
        if kind == "diag":
            diag = parse_vector(
                value.get("diagonal"), f"{field}.diagonal", dimension
            )
            return np.diag(diag)
        if kind != "full":
            raise ModelParseError(
                f"Unsupported matrix type {kind!r}",
                details={"field": field, "allowed": ["full", "diag"]},
            )
        value = value.get("data")

    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelParseError(f"{field} must be a numeric matrix") from exc

    if result.ndim != 2 or result.shape != (dimension, dimension):
        raise ModelParseError(
            f"{field} must have shape ({dimension}, {dimension})",
            details={"field": field, "actual_shape": list(result.shape)},
        )
    if not np.all(np.isfinite(result)):
        raise ModelParseError(f"{field} contains a non-finite value")

    symmetric = 0.5 * (result + result.T)
    if not np.allclose(result, symmetric, rtol=1e-9, atol=1e-9):
        asymmetry = float(np.max(np.abs(result - result.T)))
        raise ModelParseError(
            f"{field} must be symmetric",
            details={"field": field, "max_asymmetry": asymmetry},
        )

    eigenvalues = np.linalg.eigvalsh(symmetric)
    min_eigenvalue = float(eigenvalues[0])
    if min_eigenvalue < -1e-9:
        raise ModelParseError(
            f"{field} must be positive semidefinite",
            details={"field": field, "min_eigenvalue": min_eigenvalue},
        )
    return symmetric


def parse_worker(raw: Mapping[str, Any], dimension: int) -> QuadraticWorker:
    worker_id = str(raw.get("id", "")).strip()
    if not worker_id:
        raise ModelParseError("worker id is required")
    if "P" not in raw and "p" not in raw:
        raise ModelParseError(
            "worker P is required", details={"worker_id": worker_id}
        )
    if "q" not in raw:
        raise ModelParseError(
            "worker q is required", details={"worker_id": worker_id}
        )

    P = parse_matrix(
        raw.get("P", raw.get("p")), f"workers[{worker_id}].P", dimension
    )
    q = parse_vector(raw.get("q"), f"workers[{worker_id}].q", dimension)
    r = _finite_float(raw.get("r", 0.0), f"workers[{worker_id}].r")
    sample_count = raw.get("sample_count", 1)
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ModelParseError(
            "sample_count must be a positive integer",
            details={"worker_id": worker_id, "sample_count": sample_count},
        )
    return QuadraticWorker(
        worker_id=worker_id,
        P=P,
        q=q,
        r=r,
        sample_count=sample_count,
    )


def parse_problem(payload: Mapping[str, Any]) -> QuadraticModel:
    if not isinstance(payload, Mapping):
        raise ModelParseError("problem payload must be an object")

    dimension = payload.get("dimension")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ModelParseError("dimension must be a positive integer")

    raw_workers = payload.get("workers")
    if not isinstance(raw_workers, Sequence) or isinstance(
        raw_workers, (str, bytes)
    ):
        raise ModelParseError("workers must be a list")
    if not raw_workers:
        raise ModelParseError("at least one worker is required")

    workers: list[QuadraticWorker] = []
    seen: set[str] = set()
    for raw_worker in raw_workers:
        if not isinstance(raw_worker, Mapping):
            raise ModelParseError("each worker must be an object")
        worker = parse_worker(raw_worker, dimension)
        if worker.worker_id in seen:
            raise ModelParseError(
                "duplicate worker id",
                details={"worker_id": worker.worker_id},
            )
        seen.add(worker.worker_id)
        workers.append(worker)

    return QuadraticModel(dimension=dimension, workers=tuple(workers))


def model_to_json(model: QuadraticModel) -> dict[str, Any]:
    return {
        "dimension": model.dimension,
        "workers": [
            {
                "id": worker.worker_id,
                "P": worker.P.tolist(),
                "q": worker.q.tolist(),
                "r": worker.r,
                "sample_count": worker.sample_count,
            }
            for worker in model.workers
        ],
    }
