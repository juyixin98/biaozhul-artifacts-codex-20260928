"""SQLite run registry, replay events, and model-artifact storage."""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

from .errors import ComputationError, StateConflictError, ValidationError
from .model import ModelArtifact, load_artifact, save_artifact
from .training import TrainingResult


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: str
    config: dict
    corpus: list[list[str]]
    corpus_fingerprint: str
    artifact_path: str
    final_loss: float | None
    error_category: str | None
    error_message: str | None


class ModelStore:
    """Owns the SQLite connection and local artifact directory."""

    def __init__(self, db_path: str, artifact_dir: str):
        self.db_path = db_path
        self.artifact_dir = artifact_dir
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        os.makedirs(artifact_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ModelStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
                config_json TEXT NOT NULL,
                corpus_json TEXT NOT NULL,
                corpus_fingerprint TEXT NOT NULL,
                artifact_path TEXT,
                final_loss REAL,
                error_category TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                PRIMARY KEY (run_id, sequence),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.commit()

    def artifact_path_for(self, run_id: str) -> str:
        if not isinstance(run_id, str) or not run_id or "/" in run_id or os.sep in run_id:
            raise ValidationError("run_id must be a non-empty path-safe string")
        return os.path.join(self.artifact_dir, f"{run_id}.npz")

    def save_training_run(self, result: TrainingResult, corpus: list[list[str]]) -> str:
        path = self.artifact_path_for(result.run_id)
        if self.run_exists(result.run_id):
            raise StateConflictError("a run with this run_id already exists", {"run_id": result.run_id})
        try:
            with self.conn:
                save_artifact(path, result.artifact)
                self.conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, status, config_json, corpus_json,
                        corpus_fingerprint, artifact_path, final_loss
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.run_id,
                        "completed",
                        json.dumps(result.artifact.config, sort_keys=True),
                        json.dumps(corpus, ensure_ascii=False),
                        result.artifact.corpus_fingerprint,
                        path,
                        result.final_loss,
                    ),
                )
                self.conn.executemany(
                    "INSERT INTO events (run_id, sequence, event_json) VALUES (?, ?, ?)",
                    [
                        (
                            result.run_id,
                            index,
                            json.dumps(event, sort_keys=True, ensure_ascii=False),
                        )
                        for index, event in enumerate(result.events)
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise StateConflictError(
                "could not persist run because of a state conflict",
                {"run_id": result.run_id},
            ) from exc
        return path

    def record_failure(
        self,
        run_id: str,
        *,
        config: dict,
        corpus: list[list[str]],
        fingerprint: str,
        category: str,
        message: str,
        events: list[dict],
    ) -> None:
        if self.run_exists(run_id):
            raise StateConflictError("a run with this run_id already exists", {"run_id": run_id})
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs (
                    run_id, status, config_json, corpus_json, corpus_fingerprint,
                    error_category, error_message
                ) VALUES (?, 'failed', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    json.dumps(config, sort_keys=True),
                    json.dumps(corpus, ensure_ascii=False),
                    fingerprint,
                    category,
                    message,
                ),
            )
            self.conn.executemany(
                "INSERT INTO events (run_id, sequence, event_json) VALUES (?, ?, ?)",
                [
                    (run_id, index, json.dumps(event, sort_keys=True, ensure_ascii=False))
                    for index, event in enumerate(events)
                ],
            )

    def run_exists(self, run_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row is not None

    def get_run(self, run_id: str) -> RunRecord:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StateConflictError("unknown run_id", {"run_id": run_id})
        try:
            config = json.loads(row["config_json"])
            corpus = json.loads(row["corpus_json"])
        except json.JSONDecodeError as exc:
            raise ValidationError("persisted run metadata is malformed", {"run_id": run_id}) from exc
        return RunRecord(
            run_id=row["run_id"],
            status=row["status"],
            config=config,
            corpus=corpus,
            corpus_fingerprint=row["corpus_fingerprint"],
            artifact_path=row["artifact_path"],
            final_loss=row["final_loss"],
            error_category=row["error_category"],
            error_message=row["error_message"],
        )

    def load_artifact(self, run_id: str) -> ModelArtifact:
        record = self.get_run(run_id)
        if record.status != "completed" or not record.artifact_path:
            raise StateConflictError(
                "run does not have a completed model artifact",
                {"run_id": run_id, "status": record.status},
            )
        artifact = load_artifact(record.artifact_path)
        if artifact.run_id != run_id or artifact.corpus_fingerprint != record.corpus_fingerprint:
            raise StateConflictError(
                "model artifact does not match the registered run",
                {"run_id": run_id},
            )
        return artifact

    def get_events(self, run_id: str) -> list[dict]:
        self.get_run(run_id)
        rows = self.conn.execute(
            "SELECT event_json FROM events WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        try:
            return [json.loads(row["event_json"]) for row in rows]
        except json.JSONDecodeError as exc:
            raise ValidationError("persisted replay event is malformed") from exc

    def save_evaluation(self, evaluation_id: str, run_id: str, metrics: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO evaluations (evaluation_id, run_id, metrics_json) VALUES (?, ?, ?)",
                (evaluation_id, run_id, json.dumps(metrics, sort_keys=True)),
            )
