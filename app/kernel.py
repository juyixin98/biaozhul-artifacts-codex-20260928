"""Execution kernel: run the target program on a candidate and classify
the outcome into the four failure categories."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, env_fingerprint
from .models import Document, Verdict

TARGET_EXIT_CODE = 3
OTHER_EXIT_CODE = 2


@dataclass
class RunOutcome:
    verdict: Verdict
    reason: str
    exit_code: int | None
    stderr_tail: str


def program_digest(program_path: str) -> str:
    data = Path(program_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def cache_key(settings: Settings, prog_digest: str, doc: Document) -> str:
    """Cache key binds program digest + environment fingerprint + input."""
    material = "|".join([prog_digest, env_fingerprint(settings), doc.canonical()])
    return hashlib.sha256(material.encode()).hexdigest()


def classify(exit_code: int | None, timed_out: bool, stderr: str) -> RunOutcome:
    tail = stderr.strip().splitlines()[-1][:200] if stderr.strip() else ""
    if timed_out:
        return RunOutcome(
            Verdict.INDETERMINATE,
            "execution timed out; cannot determine failure category",
            None,
            tail,
        )
    if exit_code == TARGET_EXIT_CODE and stderr.startswith("TARGET_ERROR"):
        return RunOutcome(Verdict.TARGET, "reproduced the target error", exit_code, tail)
    if exit_code == 0:
        return RunOutcome(Verdict.PASS, "program exited successfully", exit_code, tail)
    return RunOutcome(
        Verdict.OTHER,
        f"program failed with a different error (exit {exit_code})",
        exit_code,
        tail,
    )


def run_candidate(settings: Settings, doc: Document) -> RunOutcome:
    """Execute the target program on `doc` and classify the result."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(doc.to_text())
        path = fh.name
    try:
        try:
            proc = subprocess.run(
                [sys.executable, settings.target_program, path],
                capture_output=True,
                text=True,
                timeout=settings.run_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return classify(None, True, "")
        return classify(proc.returncode, False, proc.stderr)
    finally:
        Path(path).unlink(missing_ok=True)
