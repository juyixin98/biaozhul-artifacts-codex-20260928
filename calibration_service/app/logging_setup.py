"""Request-scoped logging: every log line carries a request id so a call
can be traced through validation, fit, persistence, and prediction."""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [request_id=%(request_id)s] "
            "%(name)s: %(message)s"
        )
    )
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger("calibration")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"calibration.{name}")
