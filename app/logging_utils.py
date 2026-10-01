"""Structured logging helpers.

Every operational log line is a single JSON object carrying the run identity
(run_id, input_hash) plus kernel/library versions, so test and service logs
can always be correlated back to a concrete input and code version.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level),
        "event": event,
        **fields,
    }
    logger.log(level, json.dumps(payload, default=str, sort_keys=True))
