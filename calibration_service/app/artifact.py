"""SQLite-backed model artifact store.

Each artifact row binds the fitted step function to the meaning of the
input score (score_field), the library version, and training metadata.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict

from .calibrator import IsotonicCalibrator

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    model_id      TEXT NOT NULL,
    version       INTEGER NOT NULL,
    score_field   TEXT NOT NULL,
    payload       TEXT NOT NULL,
    n_train_rows  INTEGER NOT NULL,
    n_blocks      INTEGER NOT NULL,
    meta          TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (model_id, version)
)
"""


class ArtifactStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._conn() as conn:
            conn.execute(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, model_id: str, calibrator: IsotonicCalibrator, lib_version: str) -> dict:
        summary = calibrator.validation_summary
        meta = {
            "lib_version": lib_version,
            "extrapolation": calibrator.extrapolation,
            "validation": asdict(summary) if summary else None,
        }
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(version) AS v FROM artifacts WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            version = (row["v"] or 0) + 1
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    version,
                    calibrator.score_field,
                    json.dumps(calibrator.to_dict()),
                    summary.n_input if summary else 0,
                    int(calibrator.probabilities.size),
                    json.dumps(meta),
                    time.time(),
                ),
            )
        return {"model_id": model_id, "version": version, **meta}

    def load(self, model_id: str, version: int | None = None) -> tuple[IsotonicCalibrator, dict]:
        with self._conn() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT * FROM artifacts WHERE model_id = ? "
                    "ORDER BY version DESC LIMIT 1",
                    (model_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM artifacts WHERE model_id = ? AND version = ?",
                    (model_id, version),
                ).fetchone()
        if row is None:
            raise KeyError(f"artifact not found: model_id={model_id} version={version}")
        calibrator = IsotonicCalibrator.from_dict(json.loads(row["payload"]))
        meta = json.loads(row["meta"])
        meta.update(
            model_id=row["model_id"],
            version=row["version"],
            score_field=row["score_field"],
            n_train_rows=row["n_train_rows"],
            n_blocks=row["n_blocks"],
            created_at=row["created_at"],
        )
        return calibrator, meta

    def list_models(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT model_id, version, score_field, n_train_rows, n_blocks, "
                "created_at FROM artifacts ORDER BY model_id, version"
            ).fetchall()
        return [dict(r) for r in rows]
