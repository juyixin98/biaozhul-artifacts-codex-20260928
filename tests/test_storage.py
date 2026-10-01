from __future__ import annotations

from permutation_service.storage import RunStore


def test_run_store_persists_failed_interrupted_state(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    store = RunStore(db_path)
    run = store.create_run({"values": [1, 2], "group1_size": 1}, "hash-1")
    assert run["status"] == "pending"

    reopened = RunStore(db_path)
    reopened.mark_running(run["run_id"])

    restarted = RunStore(db_path)
    assert restarted.reset_stale_runs() == 1
    recovered = restarted.get_run(run["run_id"])

    assert recovered["status"] == "failed"
    assert recovered["error"]["code"] == "SERVICE_INTERRUPTED"
    assert recovered["result"] is None


def test_requested_cancel_blocks_pending_run_from_starting(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    run = store.create_run({"values": [1, 2], "group1_size": 1}, "hash-2")
    store.request_cancel(run["run_id"])

    marked = store.mark_running(run["run_id"])
    assert marked["status"] == "pending"
    assert marked["cancel_requested"] is True

