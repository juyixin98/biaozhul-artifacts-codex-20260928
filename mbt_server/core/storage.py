"""SQLite evidence storage for runs, decisions, observations, and reports."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .domain import RunRequest, RunResult, StepRecord
from .errors import ErrorCategory

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,
    scheduler TEXT NOT NULL,
    seed INTEGER NOT NULL,
    base_url TEXT NOT NULL,
    model_json TEXT NOT NULL,
    failure_category TEXT,
    failure_step INTEGER,
    reduced_from_step_count INTEGER,
    replay_script TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    transition TEXT NOT NULL,
    state_before TEXT NOT NULL,
    state_after TEXT NOT NULL,
    expected_status INTEGER NOT NULL,
    observed_status INTEGER NOT NULL,
    expected_error TEXT,
    observed_error TEXT,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    choice_reason TEXT NOT NULL,
    choice_index INTEGER NOT NULL,
    rng_draws INTEGER NOT NULL,
    rng_value INTEGER,
    plan_json TEXT NOT NULL,
    observation_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id);
"""


class EvidenceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def save_run(
        self,
        request: RunRequest,
        result: RunResult,
    ) -> int:
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO runs (
                    status, scheduler, seed, base_url, model_json,
                    failure_category, failure_step, reduced_from_step_count,
                    replay_script
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.status,
                    request.scheduler.value,
                    request.seed,
                    request.base_url,
                    json.dumps(request.model.model_dump(), ensure_ascii=False),
                    (
                        result.failure_category.value
                        if result.failure_category
                        else None
                    ),
                    result.failure_step,
                    result.reduced_from_step_count,
                    result.replay_script,
                ),
            )
            run_id = int(cursor.lastrowid)
            for record in result.steps:
                self._insert_step(run_id, record)
        return run_id

    def _insert_step(self, run_id: int, record: StepRecord) -> None:
        plan = record.plan
        self._connection.execute(
            """
            INSERT INTO steps (
                run_id, episode, step, transition, state_before, state_after,
                expected_status, observed_status, expected_error,
                observed_error, verdict, reason, choice_reason, choice_index,
                rng_draws, rng_value, plan_json, observation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                plan.episode,
                plan.step,
                plan.transition,
                plan.state_before,
                record.state_after,
                plan.expected_status,
                record.observation.status_code,
                plan.expected_error.value if plan.expected_error else None,
                (
                    record.observation.category.value
                    if record.observation.category
                    else None
                ),
                record.verdict,
                record.reason,
                plan.choice_reason,
                plan.choice_index,
                plan.rng_draws,
                plan.rng_value,
                json.dumps(plan.model_dump(), ensure_ascii=False),
                json.dumps(
                    record.observation.model_dump(), ensure_ascii=False
                ),
            ),
        )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT id, created_at, status, scheduler, seed, base_url,
                   failure_category, failure_step, reduced_from_step_count
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        run = self._connection.execute(
            "SELECT * FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            return None
        steps = self._connection.execute(
            """
            SELECT * FROM steps
            WHERE run_id = ?
            ORDER BY episode, step
            """,
            (run_id,),
        ).fetchall()
        data = dict(run)
        data["steps"] = [dict(row) for row in steps]
        return data

    def get_replay_script(self, run_id: int) -> str | None:
        row = self._connection.execute(
            "SELECT replay_script FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        return None if row is None else row["replay_script"]

    @staticmethod
    def category(value: str | None) -> ErrorCategory | None:
        return ErrorCategory(value) if value else None
