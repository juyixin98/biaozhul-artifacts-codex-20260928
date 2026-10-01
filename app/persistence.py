from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    method TEXT NOT NULL,
    status TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    n_dimensions INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT,
    failures_json TEXT NOT NULL,
    uncertainties_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    stage TEXT NOT NULL,
    elapsed_ms REAL,
    details_json TEXT NOT NULL,
    FOREIGN KEY(request_id) REFERENCES requests(request_id)
);

CREATE INDEX IF NOT EXISTS idx_request_steps_request ON request_steps(request_id);
"""


def ensure_database(path: Path | None = None) -> Path:
    db_path = path or settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA)
    return db_path


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    db_path = ensure_database(path)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def record_request(
    *,
    request_id: str,
    method: str,
    status: str,
    n_samples: int,
    n_dimensions: int,
    duration_ms: float,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None,
    failures: list[dict[str, Any]],
    uncertainties: list[dict[str, Any]],
) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO requests(
                request_id, method, status, n_samples, n_dimensions,
                duration_ms, request_json, response_json,
                failures_json, uncertainties_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                method,
                status,
                n_samples,
                n_dimensions,
                duration_ms,
                json.dumps(request_payload, ensure_ascii=False, sort_keys=True),
                None
                if response_payload is None
                else json.dumps(response_payload, ensure_ascii=False, sort_keys=True),
                json.dumps(failures, ensure_ascii=False),
                json.dumps(uncertainties, ensure_ascii=False),
            ),
        )
        connection.execute("DELETE FROM request_steps WHERE request_id = ?", (request_id,))
        steps = (response_payload or {}).get("steps", [])
        connection.executemany(
            """
            INSERT INTO request_steps(
                request_id, step_index, stage, elapsed_ms, details_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    request_id,
                    index,
                    str(step.get("step", "unknown")),
                    step.get("elapsed_ms"),
                    json.dumps(step, ensure_ascii=False, sort_keys=True),
                )
                for index, step in enumerate(steps)
            ],
        )
