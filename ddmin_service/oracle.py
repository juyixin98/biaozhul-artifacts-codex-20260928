"""Execution kernel: runs the controlled program on a candidate input
and caches the verdict.

Cache keys bind together: the program digest (argv + file bytes), an
environment fingerprint (interpreter version + declared env overrides)
and the canonical candidate JSON. Any change to one of them is a
different cache entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

from .classifier import Classification, ExecutionOutcome, classify
from .models import ClassifierConfig, StructuredInput, Verdict
from .store import Store


def program_digest(program_argv: list[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(program_argv).encode())
    for part in program_argv:
        if os.path.isfile(part):
            with open(part, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def env_fingerprint(env_overrides: dict[str, str]) -> str:
    payload = {"python": sys.version.split()[0], "overrides": sorted(env_overrides.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_key(program_sha: str, env_sha: str, candidate: StructuredInput) -> str:
    h = hashlib.sha256()
    h.update(program_sha.encode())
    h.update(env_sha.encode())
    h.update(candidate.canonical_json().encode())
    return h.hexdigest()


class Oracle:
    """Executes candidates against the controlled program, with caching."""

    def __init__(
        self,
        store: Store,
        job_id: str,
        program_argv: list[str],
        classifier_config: ClassifierConfig,
        env_overrides: dict[str, str],
    ):
        self._store = store
        self._job_id = job_id
        self._argv = program_argv
        self._config = classifier_config
        self._env = env_overrides
        self.program_sha = program_digest(program_argv)
        self.env_sha = env_fingerprint(env_overrides)
        self.live_executions = 0
        self.cache_hits = 0

    def test(self, candidate: StructuredInput) -> Classification:
        key = cache_key(self.program_sha, self.env_sha, candidate)
        cached = self._store.cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return Classification(Verdict(cached["verdict"]), "cache: " + cached["reason"])
        outcome = self._execute(candidate)
        result = classify(outcome, self._config)
        self._store.cache_put(
            key, self._job_id, result.verdict.value, result.reason,
            outcome.exit_code, outcome.duration_ms,
        )
        self.live_executions += 1
        return result

    def _execute(self, candidate: StructuredInput) -> ExecutionOutcome:
        env = dict(os.environ)
        env.update(self._env)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                self._argv,
                input=candidate.canonical_json(),
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                env=env,
            )
            elapsed = (time.monotonic() - start) * 1000
            return ExecutionOutcome(proc.returncode, proc.stderr[-2000:], False, elapsed)
        except subprocess.TimeoutExpired as exc:
            elapsed = (time.monotonic() - start) * 1000
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return ExecutionOutcome(None, stderr[-2000:], True, elapsed)
