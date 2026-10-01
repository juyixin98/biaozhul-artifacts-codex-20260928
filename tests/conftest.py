"""Pytest fixtures with run-correlated diagnostic logging."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import pytest


class RunIdentity:
    def __init__(self, test: str, run_id: str, log_path: Path):
        self.test = test
        self.run_id = run_id
        self.log_path = log_path

    def emit(self, step: str, **details):
        event = {"run_id": self.run_id, "test": self.test, "step": step}
        event.update(details)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=True) + "\n")
        logging.getLogger("tests").info(
            "test step %s",
            step,
            extra={"extra_fields": event},
        )


@pytest.fixture
def run_identity(request, tmp_path: Path):
    identity = RunIdentity(
        test=request.node.nodeid,
        run_id=f"pytest-{uuid.uuid4().hex[:12]}",
        log_path=tmp_path / "test-run.jsonl",
    )
    identity.emit("start")
    yield identity
    identity.emit("finish")
