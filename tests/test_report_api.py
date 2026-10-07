from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mbt_server.api import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(str(tmp_path / "report.sqlite3")),
        raise_server_exceptions=False,
    )


def _run_payload(order_model: dict, base_url: str, target: str) -> dict:
    return {
        "model": order_model,
        "base_url": base_url,
        "scheduler": "targeted",
        "seed": 917,
        "target_transitions": [target],
    }


def test_report_api_records_reference_pass(
    order_model: dict,
    reference_url: str,
    tmp_path,
) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/runs",
            json=_run_payload(order_model, reference_url, "reject_close_closed"),
        )
        assert response.status_code == 201
        created = response.json()
        assert created["status"] == "passed"
        assert created["failure_category"] is None
        run_id = created["run_id"]

        report = client.get(f"/runs/{run_id}").json()
        assert report["seed"] == 917
        assert report["scheduler"] == "targeted"
        assert report["status"] == "passed"
        assert len(report["steps"]) == 4
        assert report["steps"][-1]["observed_status"] == 409
        assert report["steps"][-1]["observed_error"] == "state_conflict"


def test_report_api_failure_category_and_replay(
    order_model: dict,
    buggy_url: str,
    tmp_path,
) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/runs",
            json=_run_payload(order_model, buggy_url, "reject_close_closed"),
        )
        assert response.status_code == 201
        created = response.json()
        assert created["status"] == "failed"
        assert created["failure_category"] == "state_conflict"

        report = client.get(f"/runs/{created['run_id']}").json()
        assert report["failure_step"] == 4
        assert report["steps"][-1]["reason"].startswith(
            "expected rejection HTTP 409, observed HTTP 200"
        )

        replay = client.get(f"/runs/{created['run_id']}/replay")
        assert replay.status_code == 200
        assert "python3" in replay.text
        assert "expected status" in replay.text


def test_invalid_model_is_input_error_not_computation_failure(
    order_model: dict,
    tmp_path,
) -> None:
    invalid = dict(order_model)
    invalid["initial_state"] = "missing_state"
    with _client(tmp_path) as client:
        response = client.post(
            "/models/validate",
            json=invalid,
        )
    assert response.status_code == 422
    detail = str(response.json()["detail"])
    assert "initial_state must be listed" in detail


def test_passing_run_has_no_replay_but_listing_contains_run(
    order_model: dict,
    reference_url: str,
    tmp_path,
) -> None:
    with _client(tmp_path) as client:
        created = client.post(
            "/runs",
            json=_run_payload(order_model, reference_url, "reject_close_draft"),
        ).json()
        assert created["status"] == "passed"
        assert client.get(f"/runs/{created['run_id']}/replay").status_code == 404
        runs = client.get("/runs").json()
        assert any(run["id"] == created["run_id"] for run in runs)
