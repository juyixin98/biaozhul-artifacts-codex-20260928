"""Application services coordinating validation, fitting, logging, and storage."""

from __future__ import annotations

import hashlib
from typing import Any

from .calibration import HistogramCalibrator
from .contracts import (
    AnalysisRequest,
    AnalysisResult,
    CalibrationFitRequest,
    CalibrationResult,
    CalibrationSummary,
    RawAnalysisRequest,
    RawCalibrationFitRequest,
)
from .logging_utils import JsonLineLogger, utc_now_iso
from .metrics import NumpySamples, compute_analysis
from .repository import ALGORITHM_VERSION, RunRepository, dumps


def canonical_hash(request: Any) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "request": request.model_dump(mode="json") if hasattr(request, "model_dump") else request,
    }
    return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()


def run_identifier(prefix: str, request: Any, created_at: str) -> str:
    digest = hashlib.sha256((created_at + "\n" + dumps(request)).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


class CalibrationService:
    def __init__(self, repository: RunRepository, logger: JsonLineLogger | None = None) -> None:
        self.repository = repository
        self.logger = logger or JsonLineLogger()

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        created_at = utc_now_iso()
        input_hash = canonical_hash(request)
        run_id = run_identifier("analysis", request, created_at)
        self.logger.log("analysis_started", run_id, input_hash=input_hash, n=len(request.p), method=request.method)
        try:
            samples = NumpySamples.from_contract(request)
            result = compute_analysis(
                samples,
                method=request.method,
                requested_bins=request.n_bins,
                run_id=run_id,
                explicit_edges=request.edges,
            )
            self.repository.save_success(
                run_id=run_id,
                run_type="analysis",
                input_hash=input_hash,
                created_at=created_at,
                request=request,
                result=result,
                analyses={"evaluation": result},
            )
            self.logger.log(
                "analysis_succeeded",
                run_id,
                raw_brier=result.decomposition.raw_brier,
                occupied_bins=result.occupied_bins,
                empty_bins=result.empty_bins,
                judgment="success because all requested bins and weighted terms were computed",
            )
            return result
        except Exception as exc:
            category = getattr(exc, "category", exc.__class__.__name__)
            self.repository.save_failure(
                run_id=run_id,
                run_type="analysis",
                input_hash=input_hash,
                created_at=created_at,
                request=request,
                category=category,
                message=str(exc),
            )
            self.logger.log("analysis_failed", run_id, error_category=category, message=str(exc), judgment="failure")
            raise

    def analyze_raw(self, payload: dict) -> AnalysisResult:
        raw = RawAnalysisRequest.model_validate(payload)
        created_at = utc_now_iso()
        input_hash = hashlib.sha256(dumps({"raw": raw.model_dump(mode="json")}).encode("utf-8")).hexdigest()
        run_id = run_identifier("analysis", raw, created_at)
        try:
            request = AnalysisRequest.model_validate(raw.model_dump())
            return self.analyze_with_identity(request, run_id, input_hash, created_at)
        except Exception as exc:
            category = getattr(exc, "category", exc.__class__.__name__)
            self.repository.save_failure(
                run_id=run_id,
                run_type="analysis",
                input_hash=input_hash,
                created_at=created_at,
                request=raw,
                category=category,
                message=str(exc),
            )
            self.logger.log("analysis_failed", run_id, error_category=category, message=str(exc), judgment="failure")
            raise

    def analyze_with_identity(
        self,
        request: AnalysisRequest,
        run_id: str,
        input_hash: str,
        created_at: str,
    ) -> AnalysisResult:
        self.logger.log("analysis_started", run_id, input_hash=input_hash, n=len(request.p), method=request.method)
        samples = NumpySamples.from_contract(request)
        result = compute_analysis(
            samples,
            method=request.method,
            requested_bins=request.n_bins,
            run_id=run_id,
            explicit_edges=request.edges,
        )
        self.repository.save_success(
            run_id=run_id,
            run_type="analysis",
            input_hash=input_hash,
            created_at=created_at,
            request=request,
            result=result,
            analyses={"evaluation": result},
        )
        self.logger.log(
            "analysis_succeeded",
            run_id,
            raw_brier=result.decomposition.raw_brier,
            occupied_bins=result.occupied_bins,
            empty_bins=result.empty_bins,
            judgment="success because all requested bins and weighted terms were computed",
        )
        return result

    def fit_calibration(self, request: CalibrationFitRequest) -> CalibrationResult:
        created_at = utc_now_iso()
        input_hash = canonical_hash(request)
        run_id = run_identifier("fit", request, created_at)
        self.logger.log("fit_started", run_id, input_hash=input_hash, train_n=len(request.train.p), method=request.method)
        try:
            train = NumpySamples.from_contract(request.train)
            calibrator = HistogramCalibrator.fit(
                train.p,
                train.y,
                train.w,
                method=request.method,
                n_bins=request.n_bins,
                explicit_edges=request.train_edges,
            )
            train_result = compute_analysis(
                train,
                method=request.method,
                requested_bins=request.n_bins,
                run_id=run_id + ":train",
                explicit_edges=request.train_edges,
            )

            evaluation_result = None
            transformed = None
            if request.evaluate is not None:
                evaluation = NumpySamples.from_contract(request.evaluate)
                transformed = calibrator.transform(evaluation.p).tolist()
                evaluation_request_edges = request.evaluation_edges
                evaluation_method = request.method
                evaluation_bins = request.n_bins
                if evaluation_request_edges is None:
                    evaluation_request_edges = calibrator.edges.tolist()
                    evaluation_method = "fixed"
                    evaluation_bins = len(calibrator.edges) - 1
                evaluation_result = compute_analysis(
                    NumpySamples(
                        y=evaluation.y,
                        p=calibrator.transform(evaluation.p),
                        w=evaluation.w,
                    ),
                    method=evaluation_method,
                    requested_bins=evaluation_bins,
                    run_id=run_id + ":evaluation",
                    explicit_edges=evaluation_request_edges,
                )

            summary = CalibrationSummary(
                edges=calibrator.edges.tolist(),
                bin_predicted_rates=list(calibrator.bin_predicted_rates),
                occupied_bins=train_result.occupied_bins,
                empty_bins=train_result.empty_bins,
            )
            result = CalibrationResult(
                run_id=run_id,
                calibration=summary,
                train_analysis=train_result,
                evaluation_analysis=evaluation_result,
                transformed_probabilities=transformed,
                calculation_trace=[
                    "fitted mapping is learned only from train samples",
                    "evaluation probabilities are validated in [0, 1] before transformation",
                    "evaluation statistics use transformed probabilities and frozen fitted boundaries",
                ],
            )
            self.repository.save_success(
                run_id=run_id,
                run_type="fit",
                input_hash=input_hash,
                created_at=created_at,
                request=request,
                result=result,
                analyses={"train": train_result, "evaluation": evaluation_result}
                if evaluation_result is not None
                else {"train": train_result},
                calibrator=calibrator,
            )
            self.logger.log(
                "fit_succeeded",
                run_id,
                occupied_bins=summary.occupied_bins,
                empty_bins=summary.empty_bins,
                evaluated=evaluation_result is not None,
                judgment="success because fit and evaluation stages completed separately",
            )
            return result
        except Exception as exc:
            category = getattr(exc, "category", exc.__class__.__name__)
            self.repository.save_failure(
                run_id=run_id,
                run_type="fit",
                input_hash=input_hash,
                created_at=created_at,
                request=request,
                category=category,
                message=str(exc),
            )
            self.logger.log("fit_failed", run_id, error_category=category, message=str(exc), judgment="failure")
            raise

    def fit_calibration_raw(self, payload: dict) -> CalibrationResult:
        raw = RawCalibrationFitRequest.model_validate(payload)
        created_at = utc_now_iso()
        input_hash = hashlib.sha256(dumps({"raw": raw.model_dump(mode="json")}).encode("utf-8")).hexdigest()
        run_id = run_identifier("fit", raw, created_at)
        try:
            request = CalibrationFitRequest.model_validate(raw.model_dump())
            return self.fit_with_identity(request, run_id, input_hash, created_at)
        except Exception as exc:
            category = getattr(exc, "category", exc.__class__.__name__)
            self.repository.save_failure(
                run_id=run_id,
                run_type="fit",
                input_hash=input_hash,
                created_at=created_at,
                request=raw,
                category=category,
                message=str(exc),
            )
            self.logger.log("fit_failed", run_id, error_category=category, message=str(exc), judgment="failure")
            raise

    def fit_with_identity(
        self,
        request: CalibrationFitRequest,
        run_id: str,
        input_hash: str,
        created_at: str,
    ) -> CalibrationResult:
        self.logger.log("fit_started", run_id, input_hash=input_hash, train_n=len(request.train.p), method=request.method)
        train = NumpySamples.from_contract(request.train)
        calibrator = HistogramCalibrator.fit(
            train.p,
            train.y,
            train.w,
            method=request.method,
            n_bins=request.n_bins,
            explicit_edges=request.train_edges,
        )
        train_result = compute_analysis(
            train,
            method=request.method,
            requested_bins=request.n_bins,
            run_id=run_id + ":train",
            explicit_edges=request.train_edges,
        )

        evaluation_result = None
        transformed = None
        if request.evaluate is not None:
            evaluation = NumpySamples.from_contract(request.evaluate)
            transformed_array = calibrator.transform(evaluation.p)
            transformed = transformed_array.tolist()
            evaluation_request_edges = request.evaluation_edges
            evaluation_method = request.method
            evaluation_bins = request.n_bins
            if evaluation_request_edges is None:
                evaluation_request_edges = calibrator.edges.tolist()
                evaluation_method = "fixed"
                evaluation_bins = len(calibrator.edges) - 1
            evaluation_result = compute_analysis(
                NumpySamples(y=evaluation.y, p=transformed_array, w=evaluation.w),
                method=evaluation_method,
                requested_bins=evaluation_bins,
                run_id=run_id + ":evaluation",
                explicit_edges=evaluation_request_edges,
            )

        summary = CalibrationSummary(
            edges=calibrator.edges.tolist(),
            bin_predicted_rates=list(calibrator.bin_predicted_rates),
            occupied_bins=train_result.occupied_bins,
            empty_bins=train_result.empty_bins,
        )
        result = CalibrationResult(
            run_id=run_id,
            calibration=summary,
            train_analysis=train_result,
            evaluation_analysis=evaluation_result,
            transformed_probabilities=transformed,
            calculation_trace=[
                "fitted mapping is learned only from train samples",
                "evaluation probabilities are validated in [0, 1] before transformation",
                "evaluation statistics use transformed probabilities and frozen fitted boundaries",
            ],
        )
        self.repository.save_success(
            run_id=run_id,
            run_type="fit",
            input_hash=input_hash,
            created_at=created_at,
            request=request,
            result=result,
            analyses={"train": train_result, "evaluation": evaluation_result}
            if evaluation_result is not None
            else {"train": train_result},
            calibrator=calibrator,
        )
        self.logger.log(
            "fit_succeeded",
            run_id,
            occupied_bins=summary.occupied_bins,
            empty_bins=summary.empty_bins,
            evaluated=evaluation_result is not None,
            judgment="success because fit and evaluation stages completed separately",
        )
        return result

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.repository.get_run(run_id)
