"""Application service that validates runs, invokes the kernel, and logs them."""

from __future__ import annotations

import uuid
from typing import Any

from app.config import MODEL_VERSION, SPEC_VERSION, FIXED_CONFIG
from app.errors import StlError
from app.kernel.stl import decompose_stl
from app.serialization import to_jsonable
from app.storage import RunRepository


class DecompositionService:
    def __init__(self, repository: RunRepository) -> None:
        self.repository = repository

    def decompose(
        self, run_id: str | None, values: list[float]
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        resolved_id = run_id or f"run-{uuid.uuid4().hex}"
        request = {"run_id": run_id, "values": values}

        # Check state before spending CPU. Duplicate IDs represent the same
        # replay identity and must not silently create a second result.
        self.repository.ensure_absent(resolved_id)
        try:
            result = decompose_stl(values)
            response = {
                "run_id": resolved_id,
                "trend": to_jsonable(result.trend),
                "seasonal": to_jsonable(result.seasonal),
                "residual": to_jsonable(result.residual),
                "weights": to_jsonable(result.weights),
                "diagnostics": to_jsonable(result.diagnostics),
                "trace": to_jsonable(result.trace),
            }
            self.repository.save_success(
                resolved_id,
                request,
                {
                    key: response[key]
                    for key in ("trend", "seasonal", "residual", "weights", "diagnostics")
                },
                result.trace,
                result.diagnostics["config_fingerprint"],
            )
            return resolved_id, response, result.trace
        except StlError as exc:
            self.repository.save_failure(
                resolved_id, request, exc.error_type, exc.message, exc.details
            )
            raise

    def version(self) -> dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "model_version": MODEL_VERSION,
            "config": FIXED_CONFIG,
        }

