from __future__ import annotations

import json
import logging
import uuid

from survey_backend.config import Settings
from survey_backend.models import AnalysisRequest, AnalysisResponse
from survey_backend.persistence import ResultStore
from survey_backend.statistical.analysis import run_comparison
from survey_backend.statistical.data import prepare_records
from survey_backend.statistical.privacy import redact_identifier, stable_hash


def build_request_hash(request: AnalysisRequest) -> str:
    return stable_hash(
        json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


class AnalysisService:
    def __init__(self, store: ResultStore, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.store = store
        self.settings = settings
        self.logger = logger or logging.getLogger("survey_backend")

    def analyze(
        self,
        request: AnalysisRequest,
        analysis_id: str | None,
        supplied_request_id: str | None,
    ) -> AnalysisResponse:
        request_id = supplied_request_id or request.request_id or str(uuid.uuid4())
        request_hash = build_request_hash(request)
        resolved_analysis_id = analysis_id or f"analysis-{request_hash[:16]}"
        seed = 20260930 if request.seed is None else request.seed
        confidence = self.settings.default_confidence if request.confidence is None else request.confidence
        imputations = (
            self.settings.default_imputations if request.imputations is None else request.imputations
        )
        burn_in = self.settings.default_burn_in if request.burn_in is None else request.burn_in

        self.logger.info(
            "analysis accepted request_id=%s dataset=%s records=%d",
            redact_identifier(request_id),
            request.dataset_id,
            len(request.records),
        )
        data = prepare_records(request.records)
        response = run_comparison(
            data,
            analysis_id=resolved_analysis_id,
            version=0,
            dataset_id=request.dataset_id,
            request_id=request_id,
            request_hash=request_hash,
            confidence=confidence,
            seed=seed,
            imputations=imputations,
            burn_in=burn_in,
            methods=request.methods,
        )
        saved = self.store.save(response)
        self.logger.info(
            "analysis persisted analysis_id=%s version=%d request_id=%s",
            redact_identifier(saved.analysis_id),
            saved.version,
            redact_identifier(request_id),
        )
        return saved
