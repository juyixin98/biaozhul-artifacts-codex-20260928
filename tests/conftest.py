from __future__ import annotations

from pathlib import Path

import pytest

from tests.replay_logger import ReplayLogger


LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOGGER: ReplayLogger | None = None


@pytest.fixture(autouse=True)
def replay_logger(request: pytest.FixtureRequest) -> ReplayLogger:
    logger = request.config.option.replay_logger
    logger.test_start(request.node.nodeid)
    yield logger


def pytest_configure(config: pytest.Config) -> None:
    global _LOGGER
    logger = ReplayLogger(LOGS_DIR)
    _LOGGER = logger
    config.option.replay_logger = logger
    config._beta_binomial_log_path = logger.path  # type: ignore[attr-defined]


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    terminalreporter.write_line(f"Replay log: {config._beta_binomial_log_path}")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    assert _LOGGER is not None
    if report.when == "call" or (
        report.when in {"setup", "teardown"} and report.failed
    ):
        _LOGGER.test_finish(report)


def pytest_unconfigure(config: pytest.Config) -> None:
    config.option.replay_logger.close()
