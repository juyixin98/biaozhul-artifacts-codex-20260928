"""In-memory idempotency state for the local FastAPI service."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .diagnostics import canonical_hash
from .errors import StateConflictError


@dataclass(frozen=True)
class StoredResponse:
    status_code: int
    body: dict[str, Any]


class IdempotencyStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, StoredResponse]] = {}
        self._in_flight: dict[str, str] = {}

    @staticmethod
    def payload_hash(payload: dict[str, Any]) -> str:
        return canonical_hash(payload)

    def get_or_lock(self, key: str, payload_hash: str) -> StoredResponse | None:
        with self._lock:
            existing = self._entries.get(key)
            if existing is None:
                in_flight_hash = self._in_flight.get(key)
                if in_flight_hash is not None:
                    if in_flight_hash != payload_hash:
                        raise StateConflictError(
                            "The idempotency key is already in progress for a different request.",
                            details={"idempotency_key": key},
                        )
                    raise StateConflictError(
                        "A request using this idempotency key is already in progress.",
                        details={"idempotency_key": key},
                    )
                self._in_flight[key] = payload_hash
                return None
            old_hash, _stored = existing
            if old_hash != payload_hash:
                raise StateConflictError(
                    "The idempotency key was already used with a different request.",
                    details={
                        "idempotency_key": key,
                        "existing_request_hash": old_hash,
                        "new_request_hash": payload_hash,
                    },
                )
            return _stored

    def store(self, key: str, payload_hash: str, response: StoredResponse) -> None:
        with self._lock:
            self._entries[key] = (payload_hash, response)
            self._in_flight.pop(key, None)

    def release(self, key: str, payload_hash: str) -> None:
        with self._lock:
            if self._in_flight.get(key) == payload_hash:
                self._in_flight.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._in_flight.clear()
