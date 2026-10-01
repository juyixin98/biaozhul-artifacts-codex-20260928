"""Redacted application logging."""

from __future__ import annotations

import logging


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("lagcorr")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_analysis(logger: logging.Logger, request_id: str, statuses: dict[str, int]) -> None:
    # Values are deliberately omitted. This is the safe shape for potentially
    # sensitive measurement data.
    logger.info(
        "request_id=%s analysis_complete accepted=%d rejected=%d unknown=%d",
        request_id,
        statuses.get("accepted", 0),
        statuses.get("rejected", 0),
        statuses.get("unknown", 0),
    )
