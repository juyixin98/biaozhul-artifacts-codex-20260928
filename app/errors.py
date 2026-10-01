"""Error types shared by parsing, storage and HTTP layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RequestError(Exception):
    status_code: int
    code: str
    message: str
    details: dict[str, Any] | None = None


def validation_error(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> RequestError:
    return RequestError(422, code, message, details)


def not_found_error(code: str, message: str, details: dict[str, Any] | None = None) -> RequestError:
    return RequestError(404, code, message, details)


def conflict_error(code: str, message: str, details: dict[str, Any] | None = None) -> RequestError:
    return RequestError(409, code, message, details)

