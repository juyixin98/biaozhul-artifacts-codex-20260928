"""Masking of sensitive record fields for diagnostics and logs."""
from __future__ import annotations

from typing import Any, Iterable

MASK = "***"


def mask_record(record: dict[str, Any], sensitive_fields: Iterable[str]) -> dict[str, Any]:
    fields = set(sensitive_fields)
    return {k: (MASK if k in fields else v) for k, v in record.items()}


def mask_input(records: list[dict[str, Any]], sensitive_fields: Iterable[str]) -> list[dict[str, Any]]:
    return [mask_record(r, sensitive_fields) for r in records]
