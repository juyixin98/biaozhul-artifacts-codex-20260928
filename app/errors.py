"""Domain-specific errors with stable machine-readable categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CovarianceError(Exception):
    """Base error for covariance request failures."""

    code = "covariance_error"

    def __init__(
        self,
        message: str,
        *,
        stage: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.details = details or {}


class SampleContractError(CovarianceError):
    """The supplied sample matrix does not satisfy the request contract."""

    code = "sample_contract_error"


class EstimationError(CovarianceError):
    """The estimator cannot produce a result for the supplied sample."""

    code = "estimation_error"


class MatrixValidationError(CovarianceError):
    """An estimated matrix failed an independent numerical validation."""

    code = "matrix_validation_error"


@dataclass
class FailureRecord:
    """Serializable representation of a failed request."""

    code: str
    category: str
    stage: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, error: CovarianceError) -> "FailureRecord":
        return cls(
            code=error.code,
            category=error.__class__.__name__,
            stage=error.stage,
            message=error.message,
            details=error.details,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category,
            "stage": self.stage,
            "message": self.message,
            "details": self.details,
        }
