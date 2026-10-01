import contextvars
import json
import logging
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.config import SERVICE_NAME, SERVICE_VERSION, SOLVER_NAME, log_level


request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": SERVICE_NAME,
            "service_version": SERVICE_VERSION,
            "solver": SOLVER_NAME,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "processing_location": {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_data", None)
        if extra:
            payload["data"] = extra
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("qp")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(log_level())
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, message: str, **data: Any) -> None:
    logger.log(level, message, extra={"extra_data": data, "request_id": request_id_var.get()})


def new_request_id(supplied: str | None = None) -> str:
    return supplied or f"qp-{uuid4().hex[:16]}"
