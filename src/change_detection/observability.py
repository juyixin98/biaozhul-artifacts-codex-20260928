from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in {
                "args",
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
                "run_id",
                "stack_info",
                "taskName",
                "thread",
                "threadName",
            }:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def configure_logging(level: str = "INFO", log_path: str = "logs/app.log") -> logging.Logger:
    logger = logging.getLogger("change_detection")
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False

    if not any(getattr(handler, "_change_detection_handler", False) for handler in logger.handlers):
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler._change_detection_handler = True  # type: ignore[attr-defined]
        file_handler.setFormatter(JsonLineFormatter())
        stream_handler = logging.StreamHandler()
        stream_handler._change_detection_handler = True  # type: ignore[attr-defined]
        stream_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger

