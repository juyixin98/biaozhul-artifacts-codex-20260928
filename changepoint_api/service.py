"""Application service connecting contracts, kernels, and SQLite."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any

from .config import Settings
from .errors import ChangepointError
from .kernel import KernelResult, detect_mean_changepoints
from .models import (
    DetectionRequest,
    DetectionResponse,
    NumericGuard,
    RunError,
    RunStatusResponse,
    SegmentResult,
)
from .repository import SQLiteRepository
from .versions import ALGORITHM_VERSION, PACKAGE_VERSION, TIE_BREAK_RULE

LOGGER = logging.getLogger("changepoint_api")


def canonical_input_digest(request: DetectionRequest) -> str:
    encoded = json.dumps(
        request.model_dump(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DetectionService:
    def __init__(self, settings: Settings, repository: SQLiteRepository) -> None:
        self.settings = settings
        self.repository = repository

    def _make_event_sink(self, run_id: str, input_digest: str):
        def emit(level: str, event_type: str, details: dict[str, Any]) -> None:
            event_details = {
                "run_id": run_id,
                "input_digest": input_digest,
                "package_version": PACKAGE_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
                **details,
            }
            LOGGER.log(
                getattr(logging, level),
                "run=%s digest=%s event=%s %s",
                run_id,
                input_digest[:12],
                event_type,
                details,
            )
            self.repository.append_event(
                run_id=run_id,
                level=level,
                event_type=event_type,
                message=str(
                    details.get("decision_basis")
                    or details.get("pruning_judgement")
                    or event_type
                ),
                details=event_details,
            )

        return emit

    def _response_from_result(
        self,
        *,
        run_id: str,
        input_digest: str,
        request: DetectionRequest,
        created_at: str,
        result: KernelResult,
    ) -> DetectionResponse:
        segments = [
            SegmentResult(
                start=segment.start,
                end=segment.end,
                length=segment.length,
                mean=segment.mean,
                cost=segment.cost,
                penalty=request.penalty,
            )
            for segment in result.segments
        ]
        return DetectionResponse(
            run_id=run_id,
            input_digest=input_digest,
            algorithm=request.algorithm,
            package_version=PACKAGE_VERSION,
            algorithm_version=ALGORITHM_VERSION,
            kernel_version=result.kernel_version,
            tie_break_rule=TIE_BREAK_RULE,
            n=len(request.samples),
            penalty=request.penalty,
            min_segment_length=request.min_segment_length,
            change_points=result.change_points,
            num_segments=len(segments),
            segments=segments,
            total_sse=result.total_sse,
            total_penalty=request.penalty * len(segments),
            objective=result.objective,
            numeric_guard=NumericGuard(
                prefix_strategy=(
                    "median-anchored Neumaier sums in numpy.longdouble"
                ),
                final_check="Welford and SciPy norm per emitted segment",
                prefix_anchor=result.prefix_anchor,
                max_abs_error=result.max_abs_error,
                max_relative_error=result.max_relative_error,
                clamped_negative_costs=result.clamped_negative_costs,
                candidate_evaluations=result.candidate_evaluations,
            ),
            duration_seconds=result.duration_seconds,
            created_at=created_at,
            diagnostics=result.diagnostics,
        )

    def run_detection(self, request: DetectionRequest) -> DetectionResponse:
        run_id = str(uuid.uuid4())
        input_digest = canonical_input_digest(request)
        created_at = self.repository.create_run(
            run_id=run_id,
            input_digest=input_digest,
            algorithm=request.algorithm,
            request_payload=request.model_dump(),
        )
        emit = self._make_event_sink(run_id, input_digest)
        try:
            kernel_result = detect_mean_changepoints(
                request.samples,
                penalty=request.penalty,
                min_segment_length=request.min_segment_length,
                algorithm=request.algorithm,
                max_exact_samples=self.settings.exact_dp_max_samples,
                max_pruned_samples=self.settings.pruned_dp_max_samples,
                emit=emit,
            )
            response = self._response_from_result(
                run_id=run_id,
                input_digest=input_digest,
                request=request,
                created_at=created_at,
                result=kernel_result,
            )
            payload = response.model_dump(mode="json")
            self.repository.save_result(
                run_id=run_id, result=payload, segments=payload["segments"]
            )
            emit(
                "INFO",
                "run_success",
                {
                    "objective": response.objective,
                    "change_points": response.change_points,
                    "num_segments": response.num_segments,
                    "judgement": "result persisted with status=success",
                },
            )
            return response
        except ChangepointError as exc:
            self._fail(run_id, exc.error_code, exc.message, emit)
            exc.run_id = run_id
            raise
        except Exception as exc:
            category = type(exc).__name__
            message = str(exc) or "unexpected error without a message"
            self._fail(run_id, category, message, emit)
            raise ChangepointError(
                message,
                run_id=run_id,
                status_code=500,
                error_code=category,
            ) from exc

    def _fail(self, run_id: str, category: str, message: str, emit) -> None:
        try:
            emit(
                "ERROR",
                "run_failed",
                {
                    "error_category": category,
                    "error_message": message,
                    "judgement": "result marked failed rather than returned",
                },
            )
        finally:
            self.repository.mark_failed(
                run_id=run_id, category=category, message=message
            )

    def get_run(self, run_id: str) -> RunStatusResponse:
        row = self.repository.get_run(run_id)
        if row is None:
            raise ChangepointError(
                f"run {run_id} not found",
                run_id=run_id,
                status_code=404,
                error_code="run_not_found",
            )
        result = None
        error = None
        if row["result_json"]:
            result = DetectionResponse.model_validate_json(row["result_json"])
        if row["status"] == "failed":
            error = RunError(
                category=row["error_category"],
                message=row["error_message"],
            )
        return RunStatusResponse(
            run_id=row["run_id"],
            status=row["status"],
            input_digest=row["input_digest"],
            algorithm=row["algorithm"],
            result=result,
            error=error,
            created_at=row["created_at"],
        )

    def get_events(self, run_id: str):
        rows = self.repository.get_events(run_id)
        return [
            {
                "seq": int(row["seq"]),
                "level": row["level"],
                "event_type": row["event_type"],
                "message": row["message"],
                "details": json.loads(row["details_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
