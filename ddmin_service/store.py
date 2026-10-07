"""SQLite-backed evidence store.

Three kinds of evidence:
- jobs:    specification, state machine, checkpoint (for resume), result
- queries: oracle cache, keyed by digest(program + env + candidate input)
- events:  diagnostic log with job id, request id, state and reasons
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    program_sha256 TEXT NOT NULL,
    env_sha256 TEXT NOT NULL,
    state TEXT NOT NULL,
    checkpoint_json TEXT,
    result_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS queries (
    cache_key TEXT PRIMARY KEY,
    first_job_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    exit_code INTEGER,
    duration_ms REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    request_id TEXT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    event TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- jobs -------------------------------------------------------------
    def create_job(self, job_id: str, spec_json: str, program_sha: str, env_sha: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs (job_id, created_at, spec_json, program_sha256, env_sha256, state)"
                " VALUES (?,?,?,?,?,?)",
                (job_id, utcnow(), spec_json, program_sha, env_sha, "pending"),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def update_job(
        self,
        job_id: str,
        state: str | None = None,
        checkpoint: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        sets, args = [], []
        if state is not None:
            sets.append("state=?")
            args.append(state)
        if checkpoint is not None:
            sets.append("checkpoint_json=?")
            args.append(json.dumps(checkpoint))
        if result is not None:
            sets.append("result_json=?")
            args.append(json.dumps(result))
        if error is not None:
            sets.append("error=?")
            args.append(error)
        if not sets:
            return
        args.append(job_id)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", args)

    # ---- oracle cache -------------------------------------------------------
    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM queries WHERE cache_key=?", (cache_key,)
            ).fetchone()
        return dict(row) if row else None

    def cache_put(
        self,
        cache_key: str,
        job_id: str,
        verdict: str,
        reason: str,
        exit_code: int | None,
        duration_ms: float,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO queries"
                " (cache_key, first_job_id, verdict, reason, exit_code, duration_ms, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (cache_key, job_id, verdict, reason, exit_code, duration_ms, utcnow()),
            )

    # ---- diagnostic events --------------------------------------------------
    def add_event(
        self,
        job_id: str,
        request_id: str | None,
        level: str,
        event: str,
        detail: dict[str, Any],
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO events (job_id, request_id, ts, level, event, detail_json)"
                " VALUES (?,?,?,?,?,?)",
                (job_id, request_id, utcnow(), level, event, json.dumps(detail)),
            )

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE job_id=? ORDER BY id", (job_id,)
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["detail"] = json.loads(d.pop("detail_json"))
            out.append(d)
        return out
