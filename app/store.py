"""SQLite-backed evidence store for mutation runs and mutant fates."""
from __future__ import annotations

import json
import sqlite3
import threading

from .models import MutantRecord, RunSummary

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS runs ("
    "run_id TEXT PRIMARY KEY, summary_json TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS mutants ("
    "mutant_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,"
    " record_json TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_mutants_run ON mutants(run_id);"
)


class EvidenceStore:
    """Thread-safe persistence of run summaries and mutant records."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_run(self, summary: RunSummary) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, summary_json)"
                " VALUES (?, ?)",
                (summary.run_id, summary.model_dump_json()),
            )

    def save_mutant(self, record: MutantRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO mutants(mutant_id, run_id, record_json)"
                " VALUES (?, ?, ?)",
                (record.mutant_id, record.run_id, record.model_dump_json()),
            )

    def get_run(self, run_id: str) -> RunSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return RunSummary(**json.loads(row["summary_json"])) if row else None

    def list_mutants(self, run_id: str) -> list[MutantRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_json FROM mutants WHERE run_id = ?"
                " ORDER BY mutant_id",
                (run_id,),
            ).fetchall()
        return [MutantRecord(**json.loads(r["record_json"])) for r in rows]

    def list_runs(self) -> list[RunSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT summary_json FROM runs ORDER BY created_at"
            ).fetchall()
        return [RunSummary(**json.loads(r["summary_json"])) for r in rows]
