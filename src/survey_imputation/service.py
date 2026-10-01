"""Application service that connects contracts, diagnostics, and kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid

import numpy as np

from .contracts import (
    AnalysisRequest,
    AnalysisResponse,
    GeneratedDatasetResponse,
    GenerateDatasetRequest,
    ImputedValue,
    PointResult,
    DiagnosticStatus,
)
from .diagnostics import DiagnosticContext, default_run_diagnostics
from .errors import AllOutcomeMissingError, SurveyImputationError
from .kernel import MethodEstimate, build_design, run_method
from .synthetic import generate_survey


def new_request_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass
class RejectedAnalysis(SurveyImputationError):
    category = "analysis_rejected"
    diagnostics: list = field(default_factory=list)

    def __init__(self, message: str, diagnostics: list) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class SurveyService:
    def __init__(self, redact_sensitive: bool = True) -> None:
        self.redact_sensitive = redact_sensitive

    def _context(self, request_id: str | None, prefix: str) -> DiagnosticContext:
        return DiagnosticContext(request_id or new_request_id(prefix), self.redact_sensitive)

    @staticmethod
    def _point_result(result: MethodEstimate) -> PointResult:
        return PointResult(
            method=result.method,
            model_version=result.model_version,
            n_records=result.n_records,
            n_missing=result.n_missing,
            estimate=result.estimate,
            variance=result.variance,
            standard_error=result.standard_error,
            ci95_low=result.ci95_low,
            ci95_high=result.ci95_high,
            degrees_of_freedom=result.degrees_of_freedom,
            imputation_count=result.imputation_count,
            marked_imputed=result.marked_imputed,
            note=result.note,
        )

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        context = self._context(request.request_id, "analysis")
        records = request.records
        y = np.asarray(
            [np.nan if item.rating is None else item.rating for item in records], dtype=float
        )
        covariates = np.asarray(
            [
                [item.age_z, item.member, item.used_support, item.contact_frequency]
                for item in records
            ],
            dtype=float,
        )
        mask = np.isnan(y)
        n_true_zero = int(np.nansum(y == 0.0))
        n_observed = int((~mask).sum())
        diagnostics = default_run_diagnostics(
            context,
            n=len(records),
            n_missing=int(mask.sum()),
            n_observed=n_observed,
            n_true_zero=n_true_zero,
            m=request.m if request.method == "multiple_imputation" else None,
        )
        diagnostics.append(
            context.diagnostic(
                "mar_identifiability",
                DiagnosticStatus.undetermined,
                "MAR cannot be proven from observed survey data alone. The estimators assume missingness "
                "is ignorable conditional on modeled variables; MNAR can produce non-ignorable bias.",
                {
                    "n_predictors": 4,
                    "mnar_unbiased_guaranteed": False,
                    "filled_values_are_observed": False,
                },
            )
        )

        if n_observed == 0:
            raise RejectedAnalysis(
                "all outcome ratings are missing; imputation parameters cannot be estimated",
                diagnostics,
            )
        if request.method == "complete_data" and int(mask.sum()) > 0:
            diagnostics.append(
                context.diagnostic(
                    "complete_data_requires_no_missing",
                    DiagnosticStatus.rejected,
                    "Complete-data baseline accepts only datasets without missing ratings.",
                    {"n_missing": int(mask.sum())},
                )
            )
            raise RejectedAnalysis("complete_data method received missing ratings", diagnostics)

        try:
            design = build_design(
                [item.record_id for item in records], y, covariates, provided_mask=mask
            )
            result = run_method(design, request.method, m=request.m, seed=request.seed)
        except AllOutcomeMissingError as exc:
            raise RejectedAnalysis(str(exc), diagnostics) from exc

        preview: list[ImputedValue] = []
        if result.imputed_outcomes is not None:
            missing_ids = [item.record_id for item, is_missing in zip(records, mask) if is_missing]
            for imputation_index in range(min(3, result.imputation_count or 0)):
                for record_id, value in zip(
                    missing_ids[:3], result.imputed_outcomes[imputation_index, :3]
                ):
                    preview.append(
                        ImputedValue(
                            record_id=context.redact_record_id(record_id),
                            imputation_index=imputation_index + 1,
                            rating=float(value),
                        )
                    )

        diagnostics.append(
            context.diagnostic(
                "imputed_values_not_observations",
                DiagnosticStatus.accepted,
                "Imputed values are model-based predictions used for estimation and remain marked; they are not appended as true observations.",
                {
                    "method": result.method,
                    "marked_imputed": result.marked_imputed,
                    "estimate": result.estimate,
                },
            )
        )
        return AnalysisResponse(
            request_id=context.request_id,
            estimate=self._point_result(result),
            diagnostics=diagnostics,
            imputations_preview=preview,
        )

    def generate(self, request: GenerateDatasetRequest) -> GeneratedDatasetResponse:
        context = self._context(request.request_id, "generate")
        generated = generate_survey(
            n=request.n,
            mechanism=request.mechanism,
            missing_probability=request.missing_probability,
            seed=request.seed,
            request_id=context.request_id,
        )
        for diagnostic in generated.diagnostics:
            if diagnostic.record_id is not None:
                diagnostic.record_id = context.redact_record_id(diagnostic.record_id)
        return GeneratedDatasetResponse(
            request_id=context.request_id,
            mechanism=generated.mechanism,
            seed=generated.seed,
            n_records=generated.n,
            n_missing=generated.n_missing,
            true_mean_rating=generated.true_mean_rating,
            complete_baseline_mean=generated.complete_baseline_mean,
            records=generated.records,
            diagnostics=list(generated.diagnostics),
        )
