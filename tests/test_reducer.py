"""Unit tests for ddmin. Reference answers are hand-computed here, not
produced by the implementation under test."""
from __future__ import annotations

from ddmin_service.models import Verdict
from ddmin_service.reducer import DdminState, ddmin, split_chunks, verify_one_minimal

# Hand-computed: the unique minimal failing set for this oracle is {a, b, m}.
TRIGGER = frozenset({"m", "a", "b"})
IDS = ["m", "a", "x1", "b", "x2", "x3", "x4", "x5"]


def oracle(ids):
    return Verdict.TARGET if TRIGGER <= set(ids) else Verdict.PASS


def test_split_chunks_balanced():
    assert split_chunks([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]
    assert split_chunks([1, 2, 3, 4, 5], 5) == [[1], [2], [3], [4], [5]]


def test_ddmin_finds_hand_computed_minimum():
    state, done = ddmin(IDS, oracle)
    assert done
    assert sorted(state.current_ids) == ["a", "b", "m"]


def test_result_is_one_minimal():
    state, done = ddmin(IDS, oracle)
    assert done
    ok, checks = verify_one_minimal(state.current_ids, oracle)
    assert ok
    # removing any single remaining record no longer fails
    assert all(c["verdict"] == "pass" for c in checks)
    assert sorted(c["removed"] for c in checks) == ["a", "b", "m"]


def test_resume_after_stop_matches_uninterrupted():
    calls = {"n": 0}
    budget = {"limit": 3}

    def limited(ids):
        calls["n"] += 1
        return oracle(ids)

    def should_stop():
        return calls["n"] >= budget["limit"]

    state, done = ddmin(IDS, limited, should_stop=should_stop)
    assert not done  # interrupted mid-run

    budget["limit"] = 10**9  # resume without a budget
    state, done = ddmin(IDS, limited, state=state, should_stop=should_stop)
    assert done
    assert sorted(state.current_ids) == ["a", "b", "m"]


def test_state_json_roundtrip():
    state = DdminState(["a", "b"], 4)
    assert DdminState.from_json(state.to_json()) == state
