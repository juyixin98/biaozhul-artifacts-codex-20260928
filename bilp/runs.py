"""Run registry: content-addressed run ids, idempotency, replay support.

A run id is a hash of (problem, options, solver version), so identical
requests map to the same run and any run can be replayed deterministically.
Runs live in memory and are optionally appended to a JSONL log so they
survive restarts (node archives are in-memory only).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .errors import StateConflictError
from .solver.node import NodeArchive


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def canonical_fingerprint(problem: dict, options: dict) -> str:
    return hashlib.sha256(_canonical({"problem": problem, "options": options}).encode()).hexdigest()


def make_run_id(problem: dict, options: dict, version: str) -> str:
    digest = hashlib.sha256(
        _canonical({"problem": problem, "options": options, "version": version}).encode()
    ).hexdigest()
    return f"run_{digest[:16]}"


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    problem: dict
    options: dict
    result: dict                       # full SolveResult dict, evidence included
    archive: NodeArchive | None = None  # in-memory only, for node restoration

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "problem": self.problem,
            "options": self.options,
            "result": self.result,
            "node_archive_available": self.archive is not None,
        }


class RunStore:
    def __init__(self, log_path: str | None = None):
        self._records: dict[str, RunRecord] = {}
        self._idempotency: dict[str, dict[str, str]] = {}
        self._log_path = log_path
        if log_path and os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._records[data["run_id"]] = RunRecord(
                        run_id=data["run_id"],
                        created_at=data["created_at"],
                        problem=data["problem"],
                        options=data["options"],
                        result=data["result"],
                        archive=None,
                    )

    def get(self, run_id: str) -> RunRecord | None:
        return self._records.get(run_id)

    def put(self, record: RunRecord) -> None:
        self._records[record.run_id] = record
        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "run_id": record.run_id,
                    "created_at": record.created_at,
                    "problem": record.problem,
                    "options": record.options,
                    "result": record.result,
                }) + "\n")

    def check_idempotency(self, key: str, fingerprint: str) -> RunRecord | None:
        """Return the existing run for this key, or raise on payload conflict."""
        entry = self._idempotency.get(key)
        if entry is None:
            return None
        if entry["fingerprint"] != fingerprint:
            raise StateConflictError(
                f"client_request_id {key!r} was already used with a different "
                "problem/options payload",
                {"client_request_id": key, "existing_run_id": entry["run_id"]},
            )
        return self._records.get(entry["run_id"])

    def register_idempotency(self, key: str, fingerprint: str, run_id: str) -> None:
        self._idempotency[key] = {"fingerprint": fingerprint, "run_id": run_id}


def new_run_record(run_id: str, problem: dict, options: dict,
                   result: dict, archive: NodeArchive | None) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        problem=problem,
        options=options,
        result=result,
        archive=archive,
    )
