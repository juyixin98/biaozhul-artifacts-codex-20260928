from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
from typing import Any

from app import __version__


run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", run_id_var.get()),
            "service": "bounded-trust-region-nls",
            "version": __version__,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "taskName",
                "thread",
                "threadName",
            }:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def fingerprint_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def log_event(
    logger: logging.Logger,
    event: str,
    *args: Any,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    safe_fields: dict[str, Any] = {}
    reserved = logging.makeLogRecord({}).__dict__.keys()
    for key, value in fields.items():
        safe_key = key if key not in reserved and key not in {"message", "asctime"} else f"nls_{key}"
        safe_fields[safe_key] = value
    logger.log(level, event, *args, extra=safe_fields)
