from __future__ import annotations

import pytest

from entropic_ot.errors import StateConflictError
from entropic_ot.state import IdempotencyStore, StoredResponse


def test_in_flight_idempotency_key_is_state_conflict() -> None:
    store = IdempotencyStore()
    assert store.get_or_lock("k", "hash-a") is None

    with pytest.raises(StateConflictError) as exc_info:
        store.get_or_lock("k", "hash-a")

    assert exc_info.value.status_code == 409
    assert exc_info.value.category == "state_conflict"
    store.release("k", "hash-a")
    assert store.get_or_lock("k", "hash-a") is None


def test_completed_entry_can_be_replayed() -> None:
    store = IdempotencyStore()
    assert store.get_or_lock("k", "hash-a") is None
    response = StoredResponse(200, {"ok": True})
    store.store("k", "hash-a", response)

    assert store.get_or_lock("k", "hash-a") is response

