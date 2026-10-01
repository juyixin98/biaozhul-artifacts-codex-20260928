"""Version metadata and deterministic request identifiers."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from .config import Settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def request_id_from_payload(payload: dict | list | None, provided: str | None = None) -> str:
    """Return an explicit request id or a hash based on the complete payload."""

    if provided:
        return provided.strip()
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return "req-" + hashlib.sha256(serialized.encode("ascii")).hexdigest()[:16]


def new_run_id() -> str:
    return "run-" + uuid.uuid4().hex


def version_payload(settings: Settings) -> dict:
    return {
        "service": settings.service_name,
        "api_version": settings.api_version,
        "schema_version": settings.schema_version,
        "kernel_version": settings.kernel_version,
        "algorithm_revision": settings.algorithm_revision,
        "generated_at": utc_now_iso(),
    }
