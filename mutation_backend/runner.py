"""Isolated command execution for baseline and mutant copies."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def category(self) -> str:
        if self.timed_out:
            return "timeout"
        if self.returncode == 0:
            return "passed"
        if self.returncode == 1:
            return "test_failure"
        if self.returncode in (2, 3, 4, 5):
            return "startup_failure"
        return "runner_error"


def run_test_command(
    workspace: Path,
    command: list[str],
    timeout_seconds: float,
) -> CommandResult:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(workspace) + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        elapsed = time.perf_counter() - started
        return CommandResult(
            returncode=process.returncode,
            duration_seconds=elapsed,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(
            returncode=None,
            duration_seconds=time.perf_counter() - started,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
