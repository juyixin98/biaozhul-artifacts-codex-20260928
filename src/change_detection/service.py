from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from . import SCHEMA_VERSION, __version__
from .config import Settings
from .db import RunRepository
from .dp import solve
from .errors import ChangeDetectionError

LOGGER = logging.getLogger("change_detection")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_input_hash(request: dict[str, Any]) -> str:
    encoded = json.dumps(
        request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ChangePointService:
    def __init__(self, settings: Settings | None = None, repository: RunRepository | None = None):
        self.settings = settings or Settings.from_env()
        self.repository = repository or RunRepository(self.settings.database_path)

    def analyze(self, request: dict[str, Any]) -> dict[str, Any]:
        run_id = str(uuid4())
        input_hash = canonical_input_hash(request)
        created_at = utc_now_iso()
        self.repository.create_run(
            run_id=run_id,
            schema_version=SCHEMA_VERSION,
            core_version=__version__,
            input_sha256=input_hash,
            request=request,
            created_at=created_at,
        )

        def record(event: str, payload: dict[str, Any]) -> None:
            stored_payload = {"input_sha256": input_hash, **payload}
            self.repository.append_event(run_id, event, stored_payload, utc_now_iso())
            LOGGER.info(event, extra={"run_id": run_id, **stored_payload})

        try:
            result = solve(
                request["values"],
                min_segment_length=request["min_segment_length"],
                penalty=request["penalty"],
                prune=request["prune"],
                verify_pruning=request["verify_pruning"],
                progress=record,
                run_id=run_id,
            )
            response = self.serialize_result(result, run_id=run_id, input_sha256=input_hash)
            self.repository.complete_run(run_id=run_id, result=response, updated_at=utc_now_iso())
            record(
                "run_succeeded",
                {
                    "objective": response["objective"],
                    "segment_count": response["segment_count"],
                    "breakpoints": response["breakpoints"],
                    "decision": "SUCCEEDED",
                },
            )
            return response
        except ChangeDetectionError as exc:
            self.repository.fail_run(
                run_id=run_id,
                category=exc.category,
                message=exc.message,
                updated_at=utc_now_iso(),
            )
            LOGGER.warning(
                "run_failed",
                extra={
                    "run_id": run_id,
                    "input_sha256": input_hash,
                    "category": exc.category,
                    "decision": "FAILED",
                },
            )
            raise

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.repository.get_run(run_id)

    @staticmethod
    def serialize_result(result: Any, *, run_id: str, input_sha256: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "core_version": __version__,
            "input_sha256": input_hash,
            "status": "SUCCEEDED",
            "objective": result.objective,
            "sse": result.sse,
            "penalty_per_segment": result.penalty_per_segment,
            "total_penalty": result.total_penalty,
            "segment_count": result.segment_count,
            "breakpoints": list(result.breakpoints),
            "tie_rule": result.tie_rule,
            "solution_mode": result.solution_mode,
            "verification_mode": result.verification_mode,
            "transition_count": result.transition_count,
            "pruned_transitions": result.pruned_transitions,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "mean": segment.mean,
                    "cost": segment.cost,
                }
                for segment in result.segments
            ],
        }
