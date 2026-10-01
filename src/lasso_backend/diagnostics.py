"""Request identifiers, structured diagnostics, and sensitive-data redaction."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import sparse

from .preprocessing import PreparedData
from .solver import LassoPathResult

Decision = Literal["accepted", "rejected", "undetermined"]


@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("lasso_backend")
    )

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    def add_event(
        self,
        stage: str,
        decision: Decision,
        reason: str,
        **state: Any,
    ) -> None:
        event = {
            "stage": stage,
            "decision": decision,
            "reason": reason,
            "elapsed_ms": round(self.elapsed_ms(), 3),
            "state": state,
        }
        self.events.append(event)
        level = {
            "accepted": logging.INFO,
            "rejected": logging.WARNING,
            "undetermined": logging.WARNING,
        }[decision]
        self.logger.log(
            level,
            "request_id=%s stage=%s decision=%s reason=%s",
            self.request_id,
            stage,
            decision,
            reason,
            extra={"request_id": self.request_id, "state": state},
        )


def redact_matrix(x: np.ndarray | sparse.spmatrix) -> dict[str, Any]:
    """Return metadata safe for logs without values or row-level fingerprints."""

    if sparse.issparse(x):
        x_csc = sparse.csc_matrix(x)
        return {
            "format": "sparse",
            "shape": list(x_csc.shape),
            "nnz": int(x_csc.nnz),
            "density": float(x_csc.nnz / max(x_csc.shape[0] * x_csc.shape[1], 1)),
            "has_explicit_nan": bool(np.any(~np.isfinite(x_csc.data))),
        }

    array = np.asarray(x)
    return {
        "format": "dense",
        "shape": list(array.shape),
        "nnz": int(np.count_nonzero(array)),
        "density": float(np.count_nonzero(array) / max(array.size, 1)),
        "has_explicit_nan": bool(np.any(~np.isfinite(array))),
    }


def _fingerprint(prepared: PreparedData) -> str:
    # Digest only deterministic aggregate values, never rows or observations.
    payload = {
        "n_samples": prepared.n_samples,
        "n_features": prepared.n_features,
        "zero_variance": prepared.zero_variance.tolist(),
        "scales_rms": np.sqrt(np.mean(prepared.scales**2)),
    }
    raw = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_result_diagnostic(
    context: RequestContext,
    prepared: PreparedData,
    result: LassoPathResult,
) -> dict[str, Any]:
    per_lambda = [
        {
            "lambda": item.lambda_value,
            "status": item.status,
            "decision": (
                "accepted" if item.converged else "undetermined"
            ),
            "reason": item.reason,
            "iterations": item.iterations,
            "kkt_violation": item.kkt_violation,
            "duality_gap": item.duality_gap,
            "primal_objective": item.primal_objective,
            "support_size": item.support_size,
        }
        for item in result.diagnostics
    ]
    return {
        "request_id": context.request_id,
        "decision": "accepted" if result.converged else "undetermined",
        "reason": (
            "all lambda points satisfy the KKT stopping condition"
            if result.converged
            else "one or more lambda points reached max_iter before KKT acceptance"
        ),
        "elapsed_ms": round(context.elapsed_ms(), 3),
        "data_fingerprint": _fingerprint(prepared),
        "shape": [prepared.n_samples, prepared.n_features],
        "sparse": prepared.is_sparse,
        "lambda_order": result.lambda_order,
        "warm_start": result.warm_start,
        "events": context.events,
        "per_lambda": per_lambda,
    }
