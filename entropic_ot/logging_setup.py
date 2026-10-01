"""JSONL logging with enough payload to replay a service request."""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from .diagnostics import utc_now_iso


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(
            _json_safe(payload),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("entropic_ot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = os.environ.get("ENTROPIC_OT_LOG_PATH")
    handler: logging.Handler
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(JsonLineFormatter())
    handler._entropic_ot_handler = True  # type: ignore[attr-defined]
    for existing in list(logger.handlers):
        if getattr(existing, "_entropic_ot_handler", False):
            logger.removeHandler(existing)
            existing.close()
    logger.addHandler(handler)
    return logger
