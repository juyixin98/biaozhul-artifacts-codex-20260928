from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy
import pytest
import scipy
from fastapi.testclient import TestClient

from app.api.dependencies import get_service
from app.config import Settings
from app.persistence.repository import SQLiteRepository
from app.services.beta_service import BetaBinomialService


ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-id",
        action="store",
        default=None,
        help="Reusable identifier for this test run; useful for reproducing a failure.",
    )


@pytest.fixture(scope="session")
def run_id(pytestconfig: pytest.Config) -> str:
    explicit = pytestconfig.getoption("--run-id")
    if explicit:
        value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in explicit)
    else:
        value = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
    return value


class TestRunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def note(self, name: str, stage: str, state: dict[str, Any], judgement: str) -> None:
        record = {
            "type": "note",
            "test": name,
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intermediate_state": state,
            "judgement": judgement,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


@pytest.fixture(scope="session")
def run_directory(run_id: str) -> Path:
    directory = ROOT / "test_runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def run_log_path(run_directory: Path) -> Path:
    return run_directory / "events.jsonl"


@pytest.fixture(autouse=True)
def test_logger(request: pytest.FixtureRequest, run_log_path: Path) -> TestRunLogger:
    logger = TestRunLogger(run_log_path)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    logger.note(request.node.nodeid, "started", {}, "Test execution started.")
    yield logger
    duration_seconds = time.perf_counter() - started
    outcome = "passed"
    if hasattr(request.node, "rep_call"):
        outcome = request.node.rep_call.outcome
    logger.note(
        request.node.nodeid,
        "finished",
        {"duration_seconds": duration_seconds, "started_at": started_at, "outcome": outcome},
        f"Test was recorded as {outcome}.",
    )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item):
    result = yield
    report = result.get_result()
    setattr(item, f"rep_{report.when}", report)


def _source_hashes() -> dict[str, str]:
    paths = sorted(
        path
        for path in ROOT.rglob("*.py")
        if ".pytest_cache" not in path.parts
        and "__pycache__" not in path.parts
        and "test_runs" not in path.parts
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    run_id = session.config.getoption("--run-id")
    if not run_id:
        candidates = sorted((ROOT / "test_runs").glob("run-*"), reverse=True)
        if not candidates:
            return
        run_directory = candidates[0]
    else:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in run_id)
        run_directory = ROOT / "test_runs" / safe
    manifest = {
        "run_id": run_directory.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exitstatus": exitstatus,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "httpx": httpx.__version__,
            "pytest": pytest.__version__,
        },
        "source_sha256": _source_hashes(),
        "replay_command": f"pytest -q --run-id {run_directory.name}",
        "judgement": (
            "Exit status zero means every independent assertion passed; inspect events.jsonl "
            "for the intermediate state of each test."
        ),
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


@pytest.fixture
def service(tmp_path: Path) -> BetaBinomialService:
    settings = Settings(
        database_path=tmp_path / "isolated.sqlite3",
        max_predictive_trials=10_000,
        max_observation_count=10**15,
    )
    repository = SQLiteRepository(settings.database_path)
    return BetaBinomialService(repository, settings)


@pytest.fixture
def client(service: BetaBinomialService) -> TestClient:
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def assert_near():
    def _assert_near(actual: float, expected: float, rel: float = 1e-12, abs: float = 1e-12) -> None:
        if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs):
            raise AssertionError(f"actual={actual!r} expected={expected!r}")

    return _assert_near
