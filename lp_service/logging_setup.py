"""Structured diagnostic logging helpers."""

import json
import logging
import sys
from typing import Any


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("lp_service")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def redacted_event(request_id: str, event: str, state: dict[str, Any]) -> str:
    # Requests are synthetic here, but this keeps logs free of complete model
    # vectors and records only dimensions, status and key scalar diagnostics.
    safe = {}
    for key, value in state.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[f"{key}_length"] = len(value)
        elif hasattr(value, "shape"):
            safe[f"{key}_shape"] = list(value.shape)
    return json.dumps({"request_id": request_id, "event": event, "state": safe}, sort_keys=True)
