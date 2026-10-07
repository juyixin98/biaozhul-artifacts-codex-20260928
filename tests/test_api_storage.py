from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mutation_backend.models import MutationOperator
from mutation_backend.storage import EvidenceStore


def _write_tiny_project(root: Path) -> None:
    (root / "mini").mkdir()
    (root / "mini" / "__init__.py").write_text("", encoding="utf-8")
    (root / "mini" / "math_local.py").write_text(
        "def add(a, b):\n"
        "    if a == 0:\n"
        "        return b\n"
        "    return a + b\n"
        "\n"
        "CONSTANT = 2 * 1\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_mini.py").write_text(
        "from mini.math_local import add\n"
        "\n"
        "\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
        "    assert add(0, 3) == 3\n",
        encoding="utf-8",
    )


def _payload(project: Path, test_command: list[str]) -> dict:
    return {
        "request_id": "api-mini-request",
        "project_path": str(project),
        "test_command": test_command,
        "source_paths": ["mini/**/*.py"],
        "operators": [
            MutationOperator.COMPARISON.value,
            MutationOperator.ARITHMETIC.value,
            MutationOperator.CONDITIONAL_INVERT.value,
            MutationOperator.CONDITION_ERASE.value,
        ],
        "baseline_timeout_seconds": 10,
        "mutant_timeout_seconds": 0.5,
    }


def test_api_persists_and_reports_evidence(tmp_path, monkeypatch, test_command):
    project = tmp_path / "project"
    project.mkdir()
    _write_tiny_project(project)
    monkeypatch.setenv("MUTATION_DB_PATH", str(tmp_path / "evidence.sqlite3"))
    monkeypatch.setenv("MUTATION_WORK_ROOT", str(tmp_path / "workspaces"))

    import importlib

    import mutation_backend.api as api_module

    importlib.reload(api_module)
    with TestClient(api_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["engine_version"] == "1.0.0"

        payload = _payload(project, test_command)
        payload["mutant_timeout_seconds"] = 1
        created = client.post("/mutation-runs", json=payload)
        assert created.status_code == 201
        report = created.json()
        assert report["request_id"] == "api-mini-request"
        assert report["status"] == "completed"
        assert report["source_unchanged"] is True
        assert report["summary"]["equivalent_undetermined"] >= 1
        assert report["uncertainties"]
        assert report["uncertainties"][0]["fate"] == "equivalent_undetermined"
        assert all(event["engine_version"] == "1.0.0" for event in report["events"])
        assert any(event["location"] for event in report["events"])

        fetched = client.get("/mutation-runs/api-mini-request")
        assert fetched.status_code == 200
        assert fetched.json()["summary"] == report["summary"]

        missing = client.get("/mutation-runs/does-not-exist")
        assert missing.status_code == 404


def test_evidence_store_roundtrip(tmp_path):
    from mutation_backend.models import MutationReport

    store = EvidenceStore(tmp_path / "store.sqlite3")
    payload = {
        "request_id": "roundtrip",
        "status": "completed",
        "project_path": "/tmp/project",
        "engine_version": "1.0.0",
        "baseline": {},
        "timeouts": {"baseline_seconds": 10.0, "mutant_seconds": 1.0},
        "summary": {},
        "uncertainties": [],
        "failures": [],
        "mutants": [],
        "source_before": {},
        "source_after": {},
        "source_unchanged": True,
        "events": [
            {
                "sequence": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "step": "snapshot",
                "status": "completed",
                "detail": "ok",
                "location": "/tmp/project",
                "engine_version": "1.0.0",
            }
        ],
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:01:00Z",
    }
    report = MutationReport.model_validate(payload)
    store.save_report(report)

    assert store.get_report("roundtrip") == report
    assert store.list_reports()[0]["request_id"] == "roundtrip"
