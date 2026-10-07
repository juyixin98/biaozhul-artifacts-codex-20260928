"""ddmin reduction over the record-removal operator, with checkpointable
state so an interrupted run can resume deterministically.

The oracle is assumed deterministic. When ddmin terminates, the result is
1-minimal *under the record-removal operator only*: removing any single
remaining record no longer reproduces the target failure. No claim is
made about global minimality (other operators or subsets never tested
could be smaller).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Verdict


@dataclass
class DdminState:
    current_ids: list[str]
    granularity: int = 2

    def to_json(self) -> dict:
        return {"current_ids": list(self.current_ids), "granularity": self.granularity}

    @staticmethod
    def from_json(data: dict) -> "DdminState":
        return DdminState(list(data["current_ids"]), int(data["granularity"]))


def split_chunks(items: list, n: int) -> list[list]:
    """Split items into n near-equal contiguous chunks (classic ddmin split)."""
    k, m = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(items[start : start + size])
        start += size
    return [c for c in chunks if c]


def ddmin(
    all_ids: list[str],
    test: Callable[[list[str]], Verdict],
    state: DdminState | None = None,
    on_checkpoint: Callable[[DdminState], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[DdminState, bool]:
    """Minimize all_ids while test(remaining) stays TARGET.

    Returns (state, done). When done is False (should_stop fired), the
    state can be passed back in to resume; because the oracle is
    deterministic and cached, resuming yields the same final result.
    """
    state = state or DdminState(list(all_ids), 2)
    while len(state.current_ids) >= 2:
        chunks = split_chunks(state.current_ids, state.granularity)
        reduced = False
        for chunk in chunks:
            if should_stop and should_stop():
                return state, False
            drop = set(chunk)
            complement = [i for i in state.current_ids if i not in drop]
            if not complement:
                continue
            if test(complement) == Verdict.TARGET:
                state.current_ids = complement
                state.granularity = max(state.granularity - 1, 2)
                reduced = True
                if on_checkpoint:
                    on_checkpoint(state)
                break
        if not reduced:
            if state.granularity >= len(state.current_ids):
                return state, True
            state.granularity = min(len(state.current_ids), state.granularity * 2)
            if on_checkpoint:
                on_checkpoint(state)
    return state, True


def verify_one_minimal(
    ids: list[str], test: Callable[[list[str]], Verdict]
) -> tuple[bool, list[dict[str, str]]]:
    """Check that removing any single remaining record no longer
    deterministically reproduces the target error.

    Returns (ok, checks) where checks lists each single-removal outcome.
    """
    checks = []
    ok = True
    for rid in ids:
        verdict = test([i for i in ids if i != rid])
        checks.append({"removed": rid, "verdict": verdict.value})
        if verdict == Verdict.TARGET:
            ok = False
    return ok, checks
