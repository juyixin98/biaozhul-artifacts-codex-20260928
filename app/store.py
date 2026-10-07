"""Evidence storage: SQLite-backed job records, step log, and run cache."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .models import Block, Document, Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    program_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    current_doc TEXT NOT NULL,
    pending_index INTEGER,
    steps_done INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    step_no INTEGER NOT NULL,
    action TEXT NOT NULL,
    block_text TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    candidate_digest TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS run_cache (
    cache_key TEXT PRIMARY KEY,
    program_digest TEXT NOT NULL,
    env_fingerprint TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def create_job(self, program_digest: str, doc: Document) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                (job_id, program_digest, "running", doc.canonical(), 0, 0, now, now),
            )
        return job_id

    def checkpoint(self, job_id: str, doc: Document, pending_index: int,
                   steps_done: int, status: str = "running") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET current_doc=?, pending_index=?, steps_done=?,"
                " status=?, updated_at=? WHERE job_id=?",
                (doc.canonical(), pending_index, steps_done, status,
                 time.time(), job_id),
            )

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def log_step(self, job_id: str, step_no: int, action: str,
                 block_text_masked: str, verdict: Verdict, reason: str,
                 candidate_digest: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO steps (job_id,step_no,action,block_text,verdict,"
                "reason,candidate_digest,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (job_id, step_no, action, block_text_masked, verdict.value,
                 reason, candidate_digest, time.time()),
            )

    def get_steps(self, job_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM steps WHERE job_id=? ORDER BY step_no", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def cache_get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM run_cache WHERE cache_key=?", (key,)
            ).fetchone()
        return dict(row) if row else None

    def cache_put(self, key: str, program_digest: str, env_fp: str,
                  input_digest: str, verdict: Verdict, reason: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO run_cache VALUES (?,?,?,?,?,?,?)",
                (key, program_digest, env_fp, input_digest, verdict.value,
                 reason, time.time()),
            )

    def cache_size(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) c FROM run_cache"
            ).fetchone()["c"]

    @staticmethod
    def doc_from_canonical(canonical: str) -> Document:
        pairs = json.loads(canonical)
        return Document(blocks=[Block(kind=k, text=t) for k, t in pairs])
