"""Structured logging helpers keyed by request identity."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from app.config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "details"):
            payload["details"] = record.details
        return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("covariance")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    *,
    request_id: str,
    details: dict[str, Any] | None = None,
) -> None:
    logger.log(
        level,
        message,
        extra={"event": event, "request_id": request_id, "details": details or {}},
    )
