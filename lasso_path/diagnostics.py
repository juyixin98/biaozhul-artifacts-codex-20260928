"""Structured diagnostics.

Every record answers: why was this request accepted, rejected, or left
undetermined. Records carry a ``request_id`` so a whole request can be traced.

Desensitization contract: messages and contexts may only contain shapes,
column/row indices, and aggregate statistics (norms, gaps, counts). Raw sample
values from ``X`` or ``y`` must never appear; non-scalar values are replaced
by shape summaries on construction.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Decision(str, enum.Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNDETERMINED = "undetermined"


class Severity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# Diagnostic codes (stable strings, safe to assert on in tests).
INPUT_VALIDATED = "INPUT_VALIDATED"
DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
NON_FINITE_INPUT = "NON_FINITE_INPUT"
EMPTY_INPUT = "EMPTY_INPUT"
MALFORMED_SPARSE = "MALFORMED_SPARSE"
ZERO_VARIANCE_COLUMN = "ZERO_VARIANCE_COLUMN"
ZERO_VARIANCE_DROPPED = "ZERO_VARIANCE_DROPPED"
INVALID_LAMBDA = "INVALID_LAMBDA"
INVALID_OPTION = "INVALID_OPTION"
LAMBDAS_REORDERED = "LAMBDAS_REORDERED"
LAMBDA_PATH_GENERATED = "LAMBDA_PATH_GENERATED"
NON_CONVERGENCE = "NON_CONVERGENCE"
PATH_COMPLETED = "PATH_COMPLETED"


def _sanitize(value: Any) -> Any:
    """Coerce context values to plain scalars; never leak raw array data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return f"<array shape={tuple(value.shape)}>"
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(type(value).__name__)


@dataclass
class Diagnostic:
    request_id: str
    code: str
    severity: Severity
    decision: Decision
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.context = _sanitize(self.context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "code": self.code,
            "severity": self.severity.value,
            "decision": self.decision.value,
            "message": self.message,
            "context": self.context,
        }


def info(request_id: str, code: str, message: str, **context: Any) -> Diagnostic:
    return Diagnostic(request_id, code, Severity.INFO, Decision.ACCEPTED, message, context)


def warning(request_id: str, code: str, message: str, **context: Any) -> Diagnostic:
    return Diagnostic(request_id, code, Severity.WARNING, Decision.UNDETERMINED, message, context)


def error(request_id: str, code: str, message: str, **context: Any) -> Diagnostic:
    return Diagnostic(request_id, code, Severity.ERROR, Decision.REJECTED, message, context)
