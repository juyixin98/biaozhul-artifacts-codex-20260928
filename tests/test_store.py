"""Unit tests for the evidence store and cache key binding."""
from __future__ import annotations

from ddmin_service.models import StructuredInput
from ddmin_service.oracle import cache_key, env_fingerprint
from ddmin_service.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "store.db"))


def test_cache_roundtrip(tmp_path):
    store = make_store(tmp_path)
    store.cache_put("k1", "job1", "target", "exit 3", 3, 12.5)
    row = store.cache_get("k1")
    assert row["verdict"] == "target"
    assert row["exit_code"] == 3
    assert store.cache_get("missing") is None


def test_cache_key_binds_program_env_and_input():
    candidate = StructuredInput(records=[{"id": "a", "v": 1}])
    base = cache_key("progA", "envA", candidate)
    assert cache_key("progB", "envA", candidate) != base  # program digest matters
    assert cache_key("progA", "envB", candidate) != base  # environment matters
    other = StructuredInput(records=[{"id": "a", "v": 2}])
    assert cache_key("progA", "envA", other) != base  # input matters
    assert cache_key("progA", "envA", candidate) == base  # deterministic


def test_env_fingerprint_reflects_overrides():
    assert env_fingerprint({}) != env_fingerprint({"MODE": "debug"})
    assert env_fingerprint({"A": "1", "B": "2"}) == env_fingerprint({"B": "2", "A": "1"})


def test_job_checkpoint_and_events(tmp_path):
    store = make_store(tmp_path)
    store.create_job("j1", "{}", "sha", "esha")
    store.update_job("j1", state="running", checkpoint={"current_ids": ["a"], "granularity": 2})
    job = store.get_job("j1")
    assert job["state"] == "running"
    assert '"current_ids"' in job["checkpoint_json"]
    store.add_event("j1", "req-1", "info", "run_paused", {"reason": "budget"})
    events = store.list_events("j1")
    assert events[0]["request_id"] == "req-1"
    assert events[0]["detail"]["reason"] == "budget"
