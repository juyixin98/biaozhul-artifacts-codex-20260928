"""SQLite persistence for corpora and training-run metadata.

Tables:
    corpora        validated text payloads and declared token budget
    training_runs  run lifecycle (PENDING -> COMPLETED/FAILED), config JSON,
                   artifact path/digest, stats JSON and failure envelope

The database deliberately stores metadata and text, not the numpy arrays;
those live in the hashed NPZ artifact. A completed run is immutable:
restarting it is a STATE_CONFLICT rather than a silent overwrite.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .errors import NotFoundError, StateConflictError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS corpora (
                    corpus_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    max_tokens INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    corpus_id TEXT NOT NULL REFERENCES corpora(corpus_id),
                    status TEXT NOT NULL CHECK (
                        status IN ('PENDING', 'COMPLETED', 'FAILED')
                    ),
                    config_json TEXT NOT NULL,
                    artifact_path TEXT,
                    artifact_sha256 TEXT,
                    stats_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_corpus(self, corpus_id: str, text: str, max_tokens: int | None) -> str:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO corpora(corpus_id, text, max_tokens, created_at) "
                "VALUES (?, ?, ?, ?)",
                (corpus_id, text, max_tokens, _utc_now()),
            )
        return corpus_id

    def get_corpus(self, corpus_id: str) -> sqlite3.Row:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM corpora WHERE corpus_id = ?",
                (corpus_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("corpus does not exist", corpus_id=corpus_id)
        return row

    def create_run(self, run_id: str, corpus_id: str, config: dict[str, Any]) -> None:
        # Confirm the corpus exists and keep run id insertion conflict explicit.
        self.get_corpus(corpus_id)
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT status FROM training_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                raise StateConflictError(
                    "training run id already exists",
                    run_id=run_id,
                    status=existing["status"],
                )
            now = _utc_now()
            conn.execute(
                "INSERT INTO training_runs(run_id, corpus_id, status, config_json, "
                "created_at, updated_at) VALUES (?, ?, 'PENDING', ?, ?, ?)",
                (run_id, corpus_id, json.dumps(config, sort_keys=True), now, now),
            )

    def get_run(self, run_id: str) -> sqlite3.Row:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("training run does not exist", run_id=run_id)
        return row

    def mark_completed(
        self,
        run_id: str,
        artifact_path: str,
        artifact_sha256: str,
        stats: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM training_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("training run does not exist", run_id=run_id)
            if row["status"] != "PENDING":
                raise StateConflictError(
                    "training run is not pending",
                    run_id=run_id,
                    status=row["status"],
                )
            conn.execute(
                "UPDATE training_runs SET status = 'COMPLETED', artifact_path = ?, "
                "artifact_sha256 = ?, stats_json = ?, updated_at = ? WHERE run_id = ?",
                (
                    artifact_path,
                    artifact_sha256,
                    json.dumps(stats, sort_keys=True),
                    _utc_now(),
                    run_id,
                ),
            )

    def mark_failed(self, run_id: str, error: dict[str, Any]) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM training_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("training run does not exist", run_id=run_id)
            if row["status"] != "PENDING":
                raise StateConflictError(
                    "training run is not pending",
                    run_id=run_id,
                    status=row["status"],
                )
            conn.execute(
                "UPDATE training_runs SET status = 'FAILED', error_json = ?, "
                "updated_at = ? WHERE run_id = ?",
                (json.dumps(error, sort_keys=True), _utc_now(), run_id),
            )

    def list_runs(self, corpus_id: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM training_runs"
        params: tuple[Any, ...] = ()
        if corpus_id is not None:
            query += " WHERE corpus_id = ?"
            params = (corpus_id,)
        query += " ORDER BY created_at ASC, run_id ASC"
        with self.connect() as conn:
            return list(conn.execute(query, params).fetchall())
