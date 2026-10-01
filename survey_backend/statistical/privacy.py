from __future__ import annotations

import hashlib


def redact_identifier(value: str | None) -> str | None:
    """Return a stable non-reversible label suitable for logs."""
    if value is None:
        return None
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{value[:2]}***{digest}" if len(value) > 2 else f"***{digest}"


def stable_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
