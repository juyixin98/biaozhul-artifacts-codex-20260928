"""Canonical hashes and replay-friendly evidence serialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .models import QuadraticModel
from .parser import model_to_json


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    safe_value = _json_safe(value)
    payload = canonical_json(safe_value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def model_digest(model: QuadraticModel) -> str:
    return digest(model_to_json(model))


def round_token(
    run_id: str,
    iteration: int,
    rho: float,
    z_pre: np.ndarray,
    u_pre: Mapping[str, np.ndarray],
    problem_digest: str,
) -> str:
    payload = {
        "run_id": run_id,
        "iteration": iteration,
        "rho": float(rho),
        "z_pre": np.asarray(z_pre, dtype=float).tolist(),
        "u_pre": {
            worker_id: np.asarray(value, dtype=float).tolist()
            for worker_id, value in sorted(u_pre.items())
        },
        "problem_digest": problem_digest,
    }
    return digest(payload)[:32]


def array(value: np.ndarray | Sequence[float]) -> list[float]:
    return np.asarray(value, dtype=float).tolist()
