"""Interrupt/resume: a paused run resumes from its checkpoint and reaches
the same hand-computed result as an uninterrupted run."""
from __future__ import annotations

from tests.conftest import make_spec


def test_interrupted_run_resumes_to_same_result(client):
    spec = make_spec("sum_threshold_crash.py", "overflow_input.json")

    # interrupted run first: tiny live-execution budget forces pauses
    job_id = client.post("/jobs", json=spec).json()["job_id"]
    response = client.post(f"/jobs/{job_id}/run", params={"max_live_queries": 1})
    assert response.json()["state"] == "paused"
    assert client.get(f"/jobs/{job_id}").json()["state"] == "paused"

    for _ in range(50):  # resume until done, bounded loop
        response = client.post(f"/jobs/{job_id}/run", params={"max_live_queries": 2})
        if response.json()["state"] == "done":
            break
    else:
        raise AssertionError("job did not finish after repeated resumes")

    report = client.get(f"/jobs/{job_id}/report").json()
    assert report["state"] == "done"
    assert sorted(report["minimized_record_ids"]) == ["a1", "a2", "m1"]
    assert report["one_minimal"] is True

    # uninterrupted reference run afterwards (served from the shared cache)
    ref_id = client.post("/jobs", json=spec).json()["job_id"]
    client.post(f"/jobs/{ref_id}/run")
    reference = client.get(f"/jobs/{ref_id}/report").json()
    assert reference["state"] == "done"
    assert report["minimized_record_ids"] == reference["minimized_record_ids"]

    events = client.get(f"/jobs/{job_id}/events").json()["events"]
    names = [e["event"] for e in events]
    assert "run_paused" in names and "run_resumed" in names and "run_finished" in names
