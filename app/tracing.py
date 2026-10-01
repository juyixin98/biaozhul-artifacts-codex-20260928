"""Structured trace helpers shared by computation and API layers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def trace_entry(component: str, step: str, status: str = "ok", **detail: Any) -> dict:
    return {
        "timestamp": utc_now_iso(),
        "component": component,
        "step": step,
        "status": status,
        "detail": detail,
    }


def notice(code: str, message: str, affects: str) -> dict:
    return {"code": code, "message": message, "affects": affects}


def failure(code: str, message: str, affects: str) -> dict:
    return {"code": code, "message": message, "affects": affects}
