"""Request-scoped application logging."""

from __future__ import annotations

import contextvars
import logging
import sys


request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    logger = logging.getLogger("survival")
    logger.setLevel(level)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
            )
        )
        handler.addFilter(RequestIdFilter())
        logger.addHandler(handler)
    logger.propagate = False


def get_logger() -> logging.Logger:
    return logging.getLogger("survival")
