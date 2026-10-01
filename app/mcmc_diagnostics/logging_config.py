"""Structured local logging helpers."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import Settings


def configure_logging(settings: Settings) -> logging.Logger:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mcmc_diagnostics")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s level=%(levelname)s request_id=%(request_id)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        file_handler = RotatingFileHandler(
            settings.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def log_event(
    logger: logging.Logger,
    request_id: str,
    event: str,
    level: int = logging.INFO,
    **fields: object,
) -> None:
    fields["event"] = event
    logger.log(
        level,
        json.dumps(fields, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        extra={"request_id": request_id},
    )
