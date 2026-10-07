"""Integration tests for the report interface (FastAPI TestClient)."""

import json

from fastapi.testclient import TestClient

from app.api import create_app
from app.models import Document

EXPECTED_FINAL_MASKED = "BEGIN config-v1 | mode=strict | level=5 | debug=on | END config-v1"


def make_client(settings, store):
    return TestClient(create_app(settings, store))


def sample_payload() -> dict:
    from tests.conftest import SAMPLE

    doc = Document.from_text(SAMPLE.read_text(encoding="utf-8"))
    return {"blocks": [b.model_dump() for b in doc.blocks]}


def test_full_job_lifecycle(settings, store):
    client = make_client(settings, store)
    resp = client.post("/jobs", json=sample_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["final_size"] == 5
    assert "1-minimal under the block-removal operator" in (
        body["minimality_claim"])

    job_id = body["job_id"]
    report = client.get(f"/jobs/{job_id}/report")
    assert report.status_code == 200
    rbody = report.json()
    assert rbody["final_doc_masked"] == EXPECTED_FINAL_MASKED
    assert rbody["evidence_steps"] > 0

    steps = client.get(f"/jobs/{job_id}/steps")
    assert steps.status_code == 200
    actions = {s["action"] for s in steps.json()}
    assert "remove-accepted" in actions
    assert "remove-rejected" in actions
    # every step carries a verdict and a reason (accept/reject rationale)
    for s in steps.json():
        assert s["verdict"] in ("target", "other", "pass", "indeterminate")
        assert s["reason"]


def test_sensitive_value_masked_in_api(settings, store):
    client = make_client(settings, store)
    resp = client.post("/jobs", json=sample_payload())
    job_id = resp.json()["job_id"]
    raw = json.dumps(client.get(f"/jobs/{job_id}/steps").json())
    assert "hunter2" not in raw
    assert "password=***" in raw


def test_interrupted_job_can_resume_via_api(settings, store):
    client = make_client(settings, store)
    payload = sample_payload()
    payload["max_steps"] = 1
    resp = client.post("/jobs", json=payload)
    assert resp.json()["status"] == "interrupted"
    job_id = resp.json()["job_id"]

    resp = client.post(f"/jobs/{job_id}/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"
    report = client.get(f"/jobs/{job_id}/report").json()
    assert report["final_doc_masked"] == EXPECTED_FINAL_MASKED


def test_syntax_violation_rejected(settings, store):
    client = make_client(settings, store)
    bad = {"blocks": [{"kind": "setting", "text": "a=1"},
                      {"kind": "footer", "text": "END"}]}
    resp = client.post("/jobs", json=bad)
    assert resp.status_code == 422
    assert "header" in json.dumps(resp.json()["detail"])


def test_unknown_job_404(settings, store):
    client = make_client(settings, store)
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/report").status_code == 404
    assert client.post("/jobs/nope/resume").status_code == 404
