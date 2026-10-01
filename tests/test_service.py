from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from stl_backend.errors import ResourceExhaustedError, StateConflictError
from stl_backend.service import decompose_values


def test_service_persists_result_and_replayable_iteration_states(
    tmp_path, robust_fixture
):
    db_path = tmp_path / "runs.sqlite3"
    result = decompose_values(
        robust_fixture["observed"],
        run_id="fixture-run-1",
        database_path=str(db_path),
    )

    assert result["run_id"] == "fixture-run-1"
    assert result["run_number"] == 1
    assert result["status"] == "complete"
    assert result["schema_version"] == "decomposition-response-v1"
    assert result["kernel_version"] == "constrained-stl-numpy-1.0.0"
    assert result["diagnostics"]["reconstruction_max_abs_error"] < 1e-12
    assert abs(result["diagnostics"]["seasonal_mean"]) < 1e-14
    assert len(result["iteration_history"]) == 16

    with sqlite3.connect(db_path) as conn:
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", ("fixture-run-1",)
        ).fetchone()
        events = conn.execute(
            "SELECT stage, judgment, state_json FROM run_events ORDER BY id"
        ).fetchall()

    assert run[2] == "complete"
    assert len(events) == 18
    assert events[0][0] == "contract_validation"
    assert events[1][0] == "robust_outer_iteration"
    assert events[-2][0] == "robust_outer_iteration"
    assert events[-1][0] == "acceptance"
    first_state = json.loads(events[1][2])
    assert first_state["outer_iteration"] == 0
    assert first_state["robust_weights_applied"] is False
    assert "rms_residual" in first_state
    assert "reconstruction_max_abs_error" in first_state


def test_duplicate_run_id_is_state_conflict(tmp_path, robust_fixture):
    db_path = tmp_path / "runs.sqlite3"
    first = decompose_values(
        robust_fixture["observed"],
        run_id="same-id",
        database_path=str(db_path),
    )

    with pytest.raises(StateConflictError) as exc:
        decompose_values(
            [x + 1.0 for x in robust_fixture["observed"]],
            run_id="same-id",
            database_path=str(db_path),
        )

    assert exc.value.code == "state_conflict"
    assert exc.value.details["existing_run_number"] == first["run_number"]


def test_storage_failure_is_resource_exhausted(monkeypatch, robust_fixture):
    from stl_backend import service

    class BrokenRepository:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            raise sqlite3.OperationalError("simulated disk I/O error")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(service, "RunRepository", BrokenRepository)
    with pytest.raises(ResourceExhaustedError) as exc:
        decompose_values(
            robust_fixture["observed"],
            run_id="storage-failure",
            database_path="ignored",
        )
    assert exc.value.code == "resource_exhausted"
    assert "simulated disk I/O error" in exc.value.details["database_error"]


def test_rejected_short_series_is_persisted_as_input_failure(tmp_path):
    db_path = tmp_path / "rejected.sqlite3"
    with pytest.raises(Exception):
        decompose_values(
            [1.0] * 13,
            run_id="rejected-short",
            database_path=str(db_path),
        )

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        run = conn.execute(
            "SELECT status, error_category, error_code FROM runs WHERE run_id = ?",
            ("rejected-short",),
        ).fetchone()
        event = conn.execute(
            "SELECT stage, judgment, state_json FROM run_events WHERE run_id = ?",
            ("rejected-short",),
        ).fetchone()

    assert run == ("failed", "input", "input_contract_violation")
    assert event[0] == "rejection"
    assert "input contract rejected" in event[1]
    assert "minimum_observations" in event[2]
