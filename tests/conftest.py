"""Pytest evidence: each assertion-heavy test leaves a replayable JSONL record."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

LOG_PATH = Path(__file__).resolve().parents[1] / "testlogs" / "evidence.jsonl"


@pytest.fixture
def evidence():
    data: dict = {"checks": []}

    def record(name: str, expected: object, actual: object, reason: str) -> None:
        data["checks"].append(
            {
                "check": name,
                "expected": expected,
                "actual": actual,
                "reason": reason,
            }
        )

    yield record

    data["finalized"] = True
    os.makedirs(LOG_PATH.parent, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        os.makedirs(LOG_PATH.parent, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "test_result",
                        "test": item.nodeid,
                        "run_started_ns": time.time_ns(),
                        "outcome": report.outcome,
                        "duration_seconds": round(call.duration, 9) if call else None,
                        "failure_reason": str(call.excinfo.value)
                        if call and call.excinfo
                        else None,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
