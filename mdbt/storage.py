"""SQLite evidence store.

Schema (normalised enough to answer the report API without blobs-in-blobs):

* ``runs``    - one row per campaign: seed, scheduling config, status, spec
* ``steps``   - one row per executed command, full JSON evidence per step
* ``findings``- discrepancies with category, rationale, reduction metadata,
                and the generated minimal replay script

All boundary failures are mapped to the four error categories; callers never
need SQL strings to read results back.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .errors import ComputationFailure, ResourceExhausted


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    sut_name        TEXT NOT NULL,
    sut_target      TEXT NOT NULL,
    status          TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    spec_json       TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    summary_json    TEXT NOT NULL DEFAULT '{}',
    error_category  TEXT,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    run_id      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    step_json   TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS findings (
    run_id          TEXT NOT NULL,
    finding_index   INTEGER NOT NULL,
    category        TEXT NOT NULL,
    title           TEXT NOT NULL,
    rationale       TEXT NOT NULL,
    trigger_seq     INTEGER NOT NULL,
    finding_json    TEXT NOT NULL,
    replay_script   TEXT NOT NULL DEFAULT '',
    replay_path     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, finding_index),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class EvidenceStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(self.db_path)
        except sqlite3.OperationalError as exc:
            raise ResourceExhausted(
                f"cannot open evidence database {self.db_path}: {exc}",
                detail={"db_path": self.db_path},
            ) from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def new_run_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        rand = secrets_hex(4)
        return f"run-{stamp}-{rand}"

    def create_run(
        self,
        *,
        run_id: str,
        sut_name: str,
        sut_target: str,
        seed: int,
        spec_json: dict,
        config: dict,
    ) -> None:
        try:
            self._conn.execute(
                """INSERT INTO runs
                   (run_id, created_at, sut_name, sut_target, status, seed,
                    spec_json, config_json)
                   VALUES (?, ?, ?, ?, 'running', ?, ?, ?)""",
                (
                    run_id,
                    utc_now_iso(),
                    sut_name,
                    sut_target,
                    seed,
                    json.dumps(spec_json, ensure_ascii=False),
                    json.dumps(config, ensure_ascii=False),
                ),
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            raise ResourceExhausted(
                f"cannot persist run {run_id}: {exc}", detail={"run_id": run_id}
            ) from exc

    def append_step(self, run_id: str, step: dict) -> None:
        try:
            self._conn.execute(
                "INSERT INTO steps (run_id, seq, step_json) VALUES (?, ?, ?)",
                (run_id, step["seq"], json.dumps(step, ensure_ascii=False)),
            )
        except sqlite3.OperationalError as exc:
            raise ResourceExhausted(
                f"cannot persist step {step.get('seq')}: {exc}", detail={"run_id": run_id}
            ) from exc

    def add_finding(
        self,
        *,
        run_id: str,
        index: int,
        finding_json: dict,
        replay_script: str,
        replay_path: str,
    ) -> None:
        try:
            self._conn.execute(
                """INSERT INTO findings
                   (run_id, finding_index, category, title, rationale,
                    trigger_seq, finding_json, replay_script, replay_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    index,
                    finding_json["category"],
                    finding_json["title"],
                    finding_json["rationale"],
                    finding_json["trigger_seq"],
                    json.dumps(finding_json, ensure_ascii=False),
                    replay_script,
                    replay_path,
                ),
            )
        except sqlite3.OperationalError as exc:
            raise ComputationFailure(
                f"cannot persist finding {index}: {exc}", detail={"run_id": run_id}
            ) from exc

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        summary: dict,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> None:
        try:
            self._conn.execute(
                """UPDATE runs
                   SET status = ?, summary_json = ?,
                       error_category = ?, error_message = ?
                   WHERE run_id = ?""",
                (
                    status,
                    json.dumps(summary, ensure_ascii=False),
                    error_category,
                    error_message,
                    run_id,
                ),
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            raise ResourceExhausted(
                f"cannot finish run {run_id}: {exc}", detail={"run_id": run_id}
            ) from exc

    def list_runs(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            """SELECT run_id, created_at, sut_name, sut_target, status, seed,
                      summary_json, error_category, error_message
               FROM runs ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["summary"] = json.loads(d.pop("summary_json"))
            out.append(d)
        return out

    def get_run(self, run_id: str) -> dict | None:
        r = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["spec"] = json.loads(d.pop("spec_json"))
        d["config"] = json.loads(d.pop("config_json"))
        d["summary"] = json.loads(d.pop("summary_json"))
        return d

    def get_steps(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT step_json FROM steps WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [json.loads(r["step_json"]) for r in rows]

    def get_findings(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT finding_json, replay_script, replay_path FROM findings "
            "WHERE run_id = ? ORDER BY finding_index",
            (run_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = json.loads(r["finding_json"])
            d["replay_script"] = r["replay_script"]
            d["replay_path"] = r["replay_path"]
            out.append(d)
        return out


def secrets_hex(nbytes: int) -> str:
    import secrets

    return secrets.token_hex(nbytes)
