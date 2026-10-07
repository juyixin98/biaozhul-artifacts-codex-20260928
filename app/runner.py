"""Execution kernel: runs mutants in isolated copies of the project."""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import Settings
from .models import (
    MutantOutcome,
    MutantRecord,
    RunRequest,
    RunStatus,
    RunSummary,
)
from .mutator import MutationPoint, apply_mutation, find_mutation_points
from .store import EvidenceStore

logger = logging.getLogger("mutation.runner")

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


def source_digest(root: Path) -> str:
    """Stable digest over relative paths and file contents of a project."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _iter_python_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


class MutationRunner:
    """Generates mutants and executes them against the test suite."""

    def __init__(self, settings: Settings, store: EvidenceStore):
        self._settings = settings
        self._store = store

    def _pytest(self, cwd: Path, test_dir: str, timeout: float):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [
                sys.executable, "-m", "pytest", test_dir,
                "-x", "-q", "-p", "no:cacheprovider",
            ],
            cwd=cwd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )

    def run(self, request: RunRequest) -> RunSummary:
        run_id = uuid.uuid4().hex[:12]
        src = Path(request.source_dir).resolve()
        test_dir = request.test_dir or "tests"
        log = logging.LoggerAdapter(
            logger, {"run_id": run_id, "requester": request.requester}
        )
        digest_before = source_digest(src)
        summary = RunSummary(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            requester=request.requester,
            source_dir=str(src),
            source_digest=digest_before,
            operator_set_version=self._settings.operator_set_version,
            python_version=platform.python_version(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        log.info(
            "run start backend=%s ops=%s src=%s",
            __version__, self._settings.operator_set_version, src,
        )

        work_root = self._settings.work_root or None
        workdir = Path(
            tempfile.mkdtemp(prefix=f"mutrun-{run_id}-", dir=work_root)
        )
        try:
            copy = workdir / "project"
            shutil.copytree(
                src, copy,
                ignore=shutil.ignore_patterns(*_SKIP_DIRS),
            )
            status = self._execute(
                request, run_id, src, copy, test_dir, summary, log
            )
            summary.status = status
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            digest_after = source_digest(src)
            if digest_after != digest_before:
                summary.baseline_note = (
                    "WARNING: source digest changed during run"
                )
                log.error("source directory mutated during run!")
            else:
                log.info("source digest unchanged: %s", digest_before[:12])
        self._store.save_run(summary)
        log.info("run finished status=%s", summary.status.value)
        return summary

    def _execute(self, request, run_id, src, copy, test_dir, summary, log):
        # Baseline: the unmutated suite must pass inside the copy.
        try:
            base = self._pytest(
                copy, test_dir, self._settings.baseline_timeout_seconds
            )
        except subprocess.TimeoutExpired:
            summary.baseline_note = "baseline test run exceeded timeout"
            log.warning("baseline timeout")
            return RunStatus.BASELINE_TIMEOUT
        if base.returncode != 0:
            summary.baseline_note = (
                "baseline test run failed: " + base.stdout[-500:]
            )
            log.warning("baseline failed rc=%s", base.returncode)
            return RunStatus.BASELINE_FAILED
        log.info("baseline passed")

        points: list[MutationPoint] = []
        for py in _iter_python_files(src):
            rel = py.relative_to(src)
            if rel.parts[0] == test_dir.split("/")[0]:
                continue  # never mutate the test suite itself
            try:
                source = py.read_text()
                compile(source, str(rel), "exec")
            except SyntaxError:
                log.error("original source does not compile: %s", rel)
                return RunStatus.INVALID_SOURCE
            points.extend(
                find_mutation_points(str(rel), source, set(request.operators))
            )
        points = points[: self._settings.max_mutants_per_run]
        log.info("generated %d mutation points", len(points))

        counts = {o: 0 for o in MutantOutcome}
        for seq, point in enumerate(points):
            record = self._run_mutant(
                request, run_id, seq, src, copy, test_dir, point, log
            )
            counts[record.outcome] += 1
            self._store.save_mutant(record)
        summary.total = len(points)
        summary.killed = counts[MutantOutcome.KILLED]
        summary.survived = counts[MutantOutcome.SURVIVED]
        summary.timeout = counts[MutantOutcome.TIMEOUT]
        summary.invalid_syntax = counts[MutantOutcome.INVALID_SYNTAX]
        summary.startup_failure = counts[MutantOutcome.STARTUP_FAILURE]
        summary.undetermined = counts[MutantOutcome.UNDETERMINED]
        return RunStatus.COMPLETED

    def _run_mutant(self, request, run_id, seq, src, copy, test_dir,
                    point, log) -> MutantRecord:
        mutant_id = f"{run_id}-m{seq:04d}"
        record = MutantRecord(
            mutant_id=mutant_id, run_id=run_id,
            file_path=point.file_path, line=point.line, col=point.col,
            operator=point.operator.value,
            original_segment=point.original_segment,
            mutated_segment=point.mutated_segment,
            outcome=MutantOutcome.SURVIVED,
        )
        if point.equivalent:
            record.outcome = MutantOutcome.UNDETERMINED
            record.detail = (
                "statically equivalent: mutation in unreachable code"
            )
            log.info("%s undetermined (equivalent) %s:%d",
                     mutant_id, point.file_path, point.line)
            return record

        original_source = (src / point.file_path).read_text()
        try:
            mutated_source = apply_mutation(original_source, point)
        except SyntaxError as exc:
            record.outcome = MutantOutcome.INVALID_SYNTAX
            record.detail = f"mutated source does not compile: {exc}"
            return record
        target = copy / point.file_path
        target.write_text(mutated_source)
        try:
            try:
                proc = self._pytest(copy, test_dir, request.timeout_seconds)
            except subprocess.TimeoutExpired:
                record.outcome = MutantOutcome.TIMEOUT
                record.detail = (
                    f"mutant exceeded timeout of {request.timeout_seconds}s"
                )
                log.info("%s timeout %s:%d",
                         mutant_id, point.file_path, point.line)
                return record
            if proc.returncode == 0:
                record.outcome = MutantOutcome.SURVIVED
                record.detail = "test suite passed on mutant"
            elif proc.returncode == 1:
                record.outcome = MutantOutcome.KILLED
                record.detail = "test failure: " + proc.stdout[-300:]
            else:
                record.outcome = MutantOutcome.STARTUP_FAILURE
                record.detail = (
                    f"pytest exit {proc.returncode}: " + proc.stderr[-300:]
                )
            log.info("%s %s %s:%d", mutant_id, record.outcome.value,
                     point.file_path, point.line)
            return record
        finally:
            target.write_text(original_source)  # restore copy for next mutant
