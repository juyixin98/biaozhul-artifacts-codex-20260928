"""Application service orchestrating calibration, diagnostics, and persistence."""

from __future__ import annotations

import logging
import uuid

import numpy as np

from .calibration import make_calibrator
from .config import Settings
from .contracts import (
    CalibratorInfo,
    CheckResult,
    RunRequest,
    RunResult,
    SampleSetSummary,
)
from .errors import ArithmeticConsistencyError, ContractViolation
from .kernel import KERNEL_VERSION, check_result, equal_frequency_bins, fixed_bins
from .repository import RunRepository
from .versions import deterministic_input_hash, version_info

logger = logging.getLogger("calibration.service")


def _sample_arrays(samples):
    p = np.asarray([sample.p for sample in samples], dtype=np.float64)
    y = np.asarray([sample.y for sample in samples], dtype=np.int64)
    w = np.asarray([sample.w for sample in samples], dtype=np.float64)
    return p, y, w


def _sample_summary(samples) -> SampleSetSummary:
    weights = np.asarray([sample.w for sample in samples], dtype=np.float64)
    total = float(np.sum(weights))
    ess = float(total**2 / np.sum(weights**2))
    return SampleSetSummary(
        count=len(samples),
        total_weight=total,
        effective_sample_size=ess,
        sample_ids=[sample.id for sample in samples],
    )


class CalibrationService:
    def __init__(self, settings: Settings, repository: RunRepository) -> None:
        self.settings = settings
        self.repository = repository

    def create_run(self, request: RunRequest) -> RunResult:
        run_id = request.run_id or uuid.uuid4().hex
        logger.info(
            "starting calibration run",
            extra={"extra_fields": {"run_id": run_id, "kernel": KERNEL_VERSION}},
        )

        fit_p, fit_y, fit_w = _sample_arrays(request.fit)
        eval_p, eval_y, eval_w = _sample_arrays(request.evaluation)

        fit_ids = {sample.id for sample in request.fit}
        eval_ids = {sample.id for sample in request.evaluation}
        overlap = sorted(fit_ids & eval_ids)
        if overlap:
            raise ContractViolation(
                "fit and evaluation sample ids must be disjoint",
                code="fit_eval_overlap",
                details={"overlap_ids": overlap},
            )

        logger.info(
            "fitting calibrator on fit partition only",
            extra={
                "extra_fields": {
                    "run_id": run_id,
                    "step": "fit",
                    "calibrator": request.calibrator.type,
                    "fit_n": len(request.fit),
                    "evaluation_n": len(request.evaluation),
                }
            },
        )
        calibrator = make_calibrator(request.calibrator.type, fit_p, fit_y, fit_w)
        calibrated_p = calibrator.transform(eval_p)

        logger.info(
            "evaluating fixed and equal-frequency bins",
            extra={
                "extra_fields": {
                    "run_id": run_id,
                    "step": "evaluate",
                    "fixed_n": request.fixed_bins.n_bins,
                    "quantile_n": request.quantile_bins.n_bins,
                }
            },
        )
        analyses = {
            "raw": {
                "fixed": fixed_bins(
                    eval_p,
                    eval_y,
                    eval_w,
                    n_bins=request.fixed_bins.n_bins,
                    edges=request.fixed_bins.edges,
                    tolerance=self.settings.decomposition_tolerance,
                ),
                "equal_frequency": equal_frequency_bins(
                    eval_p,
                    eval_y,
                    eval_w,
                    n_bins=request.quantile_bins.n_bins,
                    tolerance=self.settings.decomposition_tolerance,
                ),
            },
            "calibrated": {
                "fixed": fixed_bins(
                    calibrated_p,
                    eval_y,
                    eval_w,
                    n_bins=request.fixed_bins.n_bins,
                    edges=request.fixed_bins.edges,
                    tolerance=self.settings.decomposition_tolerance,
                ),
                "equal_frequency": equal_frequency_bins(
                    calibrated_p,
                    eval_y,
                    eval_w,
                    n_bins=request.quantile_bins.n_bins,
                    tolerance=self.settings.decomposition_tolerance,
                ),
            },
        }

        eval_total_weight = float(np.sum(eval_w))
        checks = [
            check_result(
                "fit_and_evaluation_samples_disjoint",
                observed=float(len(overlap)),
                expected=0.0,
                tolerance=0.0,
                details={"overlap_ids": overlap},
            ),
            check_result(
                "calibrated_probabilities_within_unit_interval",
                observed=float(np.sum((calibrated_p < 0.0) | (calibrated_p > 1.0))),
                expected=0.0,
                tolerance=0.0,
            ),
        ]

        for probability_set_name, set_results in analyses.items():
            for scheme, statistics in set_results.items():
                checks.extend(
                    [
                        statistics.weight_conservation_check,
                        statistics.count_conservation_check,
                        statistics.decomposition_check,
                    ]
                )
                raw_residual = statistics.decomposition_residual
                logger.info(
                    "completed one binning analysis",
                    extra={
                        "extra_fields": {
                            "run_id": run_id,
                            "probability_set": probability_set_name,
                            "scheme": scheme,
                            "total_weight": statistics.total_weight,
                            "expected_weight": eval_total_weight,
                            "residual": raw_residual,
                            "passed": statistics.decomposition_check.passed,
                        }
                    },
                )
                if not statistics.decomposition_check.passed:
                    raise ArithmeticConsistencyError(
                        "Brier decomposition residual exceeded configured tolerance",
                        details={
                            "probability_set": probability_set_name,
                            "scheme": scheme,
                            "residual": raw_residual,
                            "tolerance": self.settings.decomposition_tolerance,
                        },
                    )

        fingerprint_payload = request.model_dump(
            by_alias=True,
            exclude={"run_id"},
            mode="json",
        )
        input_hash = deterministic_input_hash(fingerprint_payload)
        result = RunResult(
            run_id=run_id,
            status="SUCCEEDED",
            input_hash=input_hash,
            versions=version_info(),
            calibrator=CalibratorInfo(
                type=request.calibrator.type,
                fit_summary=_sample_summary(request.fit),
                evaluation_summary=_sample_summary(request.evaluation),
                disjoint_fit_and_evaluation=not overlap,
                overlap_ids=overlap,
                parameters=calibrator.parameters(),
            ),
            analyses=analyses,
            checks=checks,
            metadata=request.metadata,
        )

        failed_checks = [check for check in checks if not check.passed]
        if failed_checks:
            raise ArithmeticConsistencyError(
                "one or more run checks failed; result is not persisted as successful",
                code="run_check_failed",
                details={"failed_checks": [check.name for check in failed_checks]},
            )

        self.repository.save(result)
        logger.info(
            "calibration run succeeded and was persisted",
            extra={"extra_fields": {"run_id": run_id, "input_hash": input_hash}},
        )
        return result

    def get_run(self, run_id: str) -> RunResult:
        return self.repository.get(run_id)

    def list_runs(self):
        return self.repository.list_summaries()
