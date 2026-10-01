from __future__ import annotations

from change_detection.db import RunRepository
from change_detection.errors import RunNotFoundError


def test_repository_persists_run_result_events_and_version(tmp_path) -> None:
    repository = RunRepository(str(tmp_path / "runs.db"))
    repository.create_run(
        run_id="run-1",
        schema_version=1,
        core_version="1.0.0",
        input_sha256="inputhash",
        request={"values": [0, 1], "min_segment_length": 1, "penalty": 0.0},
        created_at="2026-10-01T00:00:00+00:00",
    )
    repository.append_event(
        "run-1",
        "dp_progress",
        {"run_id": "run-1", "end": 2},
        "2026-10-01T00:00:01+00:00",
    )
    repository.complete_run(
        run_id="run-1",
        result={"status": "SUCCEEDED", "objective": 0.0},
        updated_at="2026-10-01T00:00:02+00:00",
    )

    stored = repository.get_run("run-1")
    assert stored["status"] == "SUCCEEDED"
    assert stored["schema_version"] == 1
    assert stored["core_version"] == "1.0.0"
    assert stored["input_sha256"] == "inputhash"
    assert stored["result"]["objective"] == 0.0
    assert stored["events"][0]["event"] == "dp_progress"
    assert stored["events"][0]["payload"]["end"] == 2


def test_repository_reports_missing_run_with_404_category(tmp_path) -> None:
    repository = RunRepository(str(tmp_path / "missing.db"))
    try:
        repository.get_run("missing")
    except RunNotFoundError as exc:
        assert exc.category == "RUN_NOT_FOUND"
        assert exc.status_code == 404
    else:
        raise AssertionError("missing run should raise RunNotFoundError")
