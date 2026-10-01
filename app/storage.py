"""SQLite-backed, versioned run store.

Each run is persisted exactly once, keyed by run_id, together with the kernel
and library versions that produced it, so any reported p-value can be traced
back and reproduced later. Cancelled runs are stored too, with a null
response payload's p_value, so they can never masquerade as completed ones.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT,
    kernel_version TEXT NOT NULL,
    library_versions_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


class RunStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_run(self, record: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs (run_id, created_at, status, input_hash,
                                     request_json, response_json, kernel_version,
                                     library_versions_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["run_id"],
                    record["created_at"],
                    record["status"],
                    record["input_hash"],
                    json.dumps(record["request"]),
                    json.dumps(record["response"]) if record.get("response") is not None else None,
                    record["kernel_version"],
                    json.dumps(record["library_versions"]),
                ),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._decode(row) if row is not None else None

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._decode(r) for r in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "input_hash": row["input_hash"],
            "request": json.loads(row["request_json"]),
            "response": json.loads(row["response_json"]) if row["response_json"] else None,
            "kernel_version": row["kernel_version"],
            "library_versions": json.loads(row["library_versions_json"]),
        }
