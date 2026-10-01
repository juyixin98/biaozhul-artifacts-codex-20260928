"""Application service around the statistical kernel."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG, DEFAULT_DATABASE_PATH, StlConfig
from .errors import (
    CalculationError,
    InputContractError,
    ResourceExhaustedError,
    StateConflictError,
)
from .kernel import validate_series, stl_decompose
from .repository import RunRepository


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_fingerprint(values: Sequence[float]) -> str:
    return hashlib.sha256(
        _json_text({"values": [float(v) for v in values]}).encode("utf-8")
    ).hexdigest()


def _invalid_request_record(
    values: Any, error: InputContractError
) -> tuple[int, str, dict[str, Any]]:
    observations = 0
    safe_values: list[Any] = []
    if isinstance(values, (list, tuple)):
        observations = len(values)
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                safe_values.append(None)
                continue
            safe_values.append(number if np.isfinite(number) else None)

    payload = {
        "values": safe_values,
        "rejection": "non_finite_or_non_numeric_observations_are_not_stored_as_zero",
    }
    payload_text = _json_text(payload)
    return (
        observations,
        hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
        payload,
    )


def _record_invalid_contract(
    *,
    run_id: str,
    values: Any,
    error: InputContractError,
    config: StlConfig,
    database_path: str,
) -> None:
    observations, fingerprint, payload = _invalid_request_record(values, error)
    with RunRepository(database_path) as repository:
        existing = repository.get_run(run_id)
        if existing is not None:
            raise StateConflictError(
                "run_id has already been used",
                details={
                    "run_id": run_id,
                    "existing_run_number": existing["id"],
                    "existing_status": existing["status"],
                },
            )
        run_number = repository.create_run(
            run_id=run_id,
            observations=observations,
            request_fingerprint=fingerprint,
            config=config.as_public_dict(),
            request_payload=payload,
        )
        repository.add_event(
            run_id=run_id,
            run_number=run_number,
            stage="rejection",
            judgment="input contract rejected before decomposition",
            state={
                "reason": error.message,
                "details": error.details,
                "observations": observations,
                "request_fingerprint": fingerprint,
            },
        )
        repository.fail_run(
            run_number=run_number,
            category="input",
            code=error.code,
            message=error.message,
        )


def _public_result(
    *,
    run_id: str,
    run_number: int,
    values: np.ndarray,
    output: dict[str, Any],
    config: StlConfig,
) -> dict[str, Any]:
    residual = np.asarray(output["residual"])
    weights = np.asarray(output["robust_weights"])
    reconstruction = (
        output["trend"] + output["seasonal"] + output["residual"]
    )
    diagnostics = {
        "observations": int(values.size),
        "seasonal_mean": float(np.mean(output["seasonal"])),
        "residual_mean": float(np.mean(residual)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "reconstruction_max_abs_error": float(
            np.max(np.abs(values - reconstruction))
        ),
        "min_robust_weight": float(np.min(weights)),
        "zero_weight_count": int(np.sum(weights == 0.0)),
        "outer_iterations_recorded": len(output["history"]),
        "centering_shift": float(output["centering_shift"]),
    }
    return {
        "run_id": run_id,
        "run_number": run_number,
        "status": "complete",
        "schema_version": config.schema_version,
        "kernel_version": output["kernel_version"],
        "config": config.as_public_dict(),
        "observed": [float(x) for x in values],
        "components": {
            "trend": [float(x) for x in output["trend"]],
            "seasonal": [float(x) for x in output["seasonal"]],
            "residual": [float(x) for x in output["residual"]],
        },
        "robust_weights": [float(x) for x in weights],
        "diagnostics": diagnostics,
        "iteration_history": output["history"],
    }


def decompose_values(
    values: Sequence[float],
    *,
    run_id: str | None = None,
    config: StlConfig = DEFAULT_CONFIG,
    database_path: str = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """Validate, decompose, persist replay metadata, and return the contract."""

    chosen_run_id = run_id or str(uuid.uuid4())
    if not isinstance(chosen_run_id, str) or not chosen_run_id.strip():
        raise InputContractError("run_id must be a non-empty string")
    if len(chosen_run_id) > 200:
        raise InputContractError("run_id must contain at most 200 characters")

    try:
        y = validate_series(values, config)
    except InputContractError as error:
        try:
            _record_invalid_contract(
                run_id=chosen_run_id,
                values=values,
                error=error,
                config=config,
                database_path=database_path,
            )
        except StateConflictError:
            raise
        except sqlite3.Error as storage_error:
            raise ResourceExhaustedError(
                "run storage is unavailable or exhausted while recording rejection",
                details={"database_error": str(storage_error)},
            ) from storage_error
        raise

    request_payload = {"values": [float(x) for x in y]}
    fingerprint = hashlib.sha256(
        _json_text(request_payload).encode("utf-8")
    ).hexdigest()

    run_number: int | None = None
    try:
        with RunRepository(database_path) as repository:
            existing = repository.get_run(chosen_run_id)
            if existing is not None:
                raise StateConflictError(
                    "run_id has already been used",
                    details={
                        "run_id": chosen_run_id,
                        "existing_run_number": existing["id"],
                        "existing_status": existing["status"],
                    },
                )

            run_number = repository.create_run(
                run_id=chosen_run_id,
                observations=int(y.size),
                request_fingerprint=fingerprint,
                config=config.as_public_dict(),
                request_payload=request_payload,
            )
            repository.add_event(
                run_id=chosen_run_id,
                run_number=run_number,
                stage="contract_validation",
                judgment="accepted finite numeric one-dimensional sample",
                state={
                    "observations": int(y.size),
                    "minimum_observations": config.minimum_observations,
                    "maximum_observations": config.max_observations,
                    "request_fingerprint": fingerprint,
                },
            )

            try:
                output = stl_decompose(y, config, collect_history=True)
                for state in output["history"]:
                    iteration = int(state["outer_iteration"])
                    repository.add_event(
                        run_id=chosen_run_id,
                        run_number=run_number,
                        iteration=iteration,
                        stage="robust_outer_iteration",
                        judgment=(
                            "weights applied before inner loop"
                            if state["robust_weights_applied"]
                            else "initial uniform-weight inner loop"
                        ),
                        state=state,
                    )

                result = _public_result(
                    run_id=chosen_run_id,
                    run_number=run_number,
                    values=y,
                    output=output,
                    config=config,
                )
                repository.add_event(
                    run_id=chosen_run_id,
                    run_number=run_number,
                    stage="acceptance",
                    judgment=(
                        "accepted finite components and additive reconstruction"
                    ),
                    state=result["diagnostics"],
                )
                repository.complete_run(
                    run_number=run_number,
                    kernel_version=config.kernel_version,
                    schema_version=config.schema_version,
                    result=result,
                )
                return result
            except Exception as exc:
                if isinstance(exc, (InputContractError, ResourceExhaustedError)):
                    category = "input"
                elif isinstance(exc, CalculationError):
                    category = "calculation"
                else:
                    category = "calculation"
                repository.add_event(
                    run_id=chosen_run_id,
                    run_number=run_number,
                    stage="rejection",
                    judgment="decomposition failed and was not returned",
                    state={"exception_type": type(exc).__name__, "reason": str(exc)},
                )
                repository.fail_run(
                    run_number=run_number,
                    category=category,
                    code=getattr(exc, "code", "calculation_failed"),
                    message=str(exc),
                )
                raise
    except sqlite3.Error as exc:
        raise ResourceExhaustedError(
            "run storage is unavailable or exhausted",
            details={"run_id": chosen_run_id, "database_error": str(exc)},
        ) from exc
