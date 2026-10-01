"""Structured diagnostic records with stable identifiers."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from .contracts import Diagnostic, DiagnosticStatus

LOGGER = logging.getLogger("survey_imputation")


@dataclass(frozen=True)
class DiagnosticContext:
    request_id: str
    redact: bool = True

    def redact_record_id(self, record_id: str | None) -> str | None:
        if record_id is None:
            return None
        if not self.redact or len(record_id) <= 6:
            return "***"
        return f"{record_id[:2]}***{record_id[-2:]}"

    def diagnostic(
        self,
        code: str,
        status: DiagnosticStatus,
        message: str,
        state: dict[str, int | float | str | bool | None],
        record_id: str | None = None,
    ) -> Diagnostic:
        safe_state = dict(state)
        if "record_id" in safe_state:
            safe_state["record_id"] = self.redact_record_id(str(safe_state["record_id"]))
        diagnostic = Diagnostic(
            code=code,
            status=status,
            message=message,
            observed_state=safe_state,
            record_id=self.redact_record_id(record_id),
        )
        LOGGER.info(
            "request=%s code=%s status=%s state=%s message=%s",
            self.request_id,
            code,
            status.value,
            safe_state,
            message,
        )
        return diagnostic


def default_run_diagnostics(
    context: DiagnosticContext,
    n: int,
    n_missing: int,
    n_observed: int,
    n_true_zero: int,
    m: int | None = None,
) -> list[Diagnostic]:
    diagnostics = [
        context.diagnostic(
            "dataset_non_empty",
            DiagnosticStatus.accepted if n > 0 else DiagnosticStatus.rejected,
            "At least one record is required." if n == 0 else "Dataset contains records.",
            {"n_records": n},
        ),
        context.diagnostic(
            "observed_training_rows",
            DiagnosticStatus.accepted if n_observed >= 3 else DiagnosticStatus.rejected,
            "At least five observed rows are required for the intercept and four predictors."
            if n_observed < 5
            else "Observed rows are available for training parameters.",
            {"n_observed": n_observed, "required": 5},
        ),
        context.diagnostic(
            "all_outcome_missing",
            DiagnosticStatus.rejected if n_observed == 0 else DiagnosticStatus.accepted,
            "No observed rating exists; neither mean nor regression parameters can be estimated."
            if n_observed == 0
            else "At least one observed rating exists.",
            {"n_observed": n_observed, "n_missing": n_missing},
        ),
        context.diagnostic(
            "structural_zeros_separated",
            DiagnosticStatus.accepted,
            "Exact zero ratings are treated as observed structural zeros, not missing values.",
            {"n_true_zero": n_true_zero},
        ),
    ]
    if m is not None:
        diagnostics.append(
            context.diagnostic(
                "multiple_imputation_count",
                DiagnosticStatus.accepted if m >= 2 else DiagnosticStatus.rejected,
                "Rubin pooling requires at least two independent imputed datasets."
                if m < 2
                else "Multiple imputations will be pooled with Rubin's rules.",
                {"m": m, "required": 2},
            )
        )
    return diagnostics
