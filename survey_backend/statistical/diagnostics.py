from __future__ import annotations

from survey_backend.models import Diagnostic
from survey_backend.statistical.data import AnalysisData
from survey_backend.statistical.privacy import redact_identifier


def make_diagnostic(
    state: str,
    code: str,
    message: str,
    observed_state: dict[str, int | float | bool | str | None],
    record_id: str | None = None,
    request_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        record_id=record_id,
        redacted_record_id=redact_identifier(record_id),
        request_id=request_id,
        redacted_request_id=redact_identifier(request_id),
        state=state,  # type: ignore[arg-type]
        code=code,
        message=message,
        observed_state=observed_state,
    )


def request_diagnostics(
    data: AnalysisData,
    request_id: str,
    missing_counts: dict[str, int],
    zero_counts: dict[str, int],
    allowed_n: int,
) -> list[Diagnostic]:
    diagnostics = [
        make_diagnostic(
            "accepted",
            "eligible_scope",
            "Estimation is restricted to records marked eligible; ineligible rows never train parameters.",
            {
                "eligible_observations": allowed_n,
                "excluded_ineligible": data.n - allowed_n,
            },
            request_id=request_id,
        )
    ]
    for field, missing_n in missing_counts.items():
        zero_n = zero_counts[field]
        diagnostics.append(
            make_diagnostic(
                "accepted",
                "missing_distinguished_from_zero",
                "Null values form the missing mask and numeric zeros remain genuine observations.",
                {
                    "field": field,
                    "missing_among_eligible": missing_n,
                    "genuine_zero_among_observed_eligible": zero_n,
                },
                request_id=request_id,
            )
        )
    high_missing = [
        field
        for field, missing_n in missing_counts.items()
        if allowed_n > 0 and missing_n / allowed_n > 0.40
    ]
    if high_missing:
        diagnostics.append(
            make_diagnostic(
                "undetermined",
                "high_missing_rate",
                "A field exceeds 40% missing among allowed observations; estimates can be sensitive to assumptions.",
                {"fields": ",".join(high_missing), "threshold": 0.40},
                request_id=request_id,
            )
        )
    diagnostics.append(
        make_diagnostic(
            "undetermined",
            "missingness_mechanism_unobserved",
            "The observed data cannot distinguish MAR from MNAR. Results assume MAR/MCAR and may be biased under MNAR.",
            {"assumption": "MAR for imputation; MCAR for complete-case unbiasedness"},
            request_id=request_id,
        )
    )
    return diagnostics


def record_warning(
    data: AnalysisData,
    index: int,
    state: str,
    code: str,
    message: str,
    observed_state: dict[str, int | float | bool | str | None],
    request_id: str,
) -> Diagnostic:
    return make_diagnostic(
        state,
        code,
        message,
        observed_state,
        record_id=data.record_ids[index],
        request_id=request_id,
    )
