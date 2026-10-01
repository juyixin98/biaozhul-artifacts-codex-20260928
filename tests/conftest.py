"""Shared fixtures: app client, solver config, replayable run logging."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bilp.config import SolverConfig
from bilp.main import create_app


@pytest.fixture
def solver_config() -> SolverConfig:
    return SolverConfig(
        max_nodes=10_000,
        time_limit_s=10.0,
        max_variables=50,
        max_constraints=100,
        max_evidence_nodes=500,
    )


@pytest.fixture
def client(solver_config):
    return TestClient(create_app(solver_config))


@pytest.fixture
def run_logger(request):
    """Append replayable run records to test-logs/session.jsonl.

    Each record keeps the run id (replays the problem deterministically),
    the problem itself, key intermediate states (per-node decisions) and
    the recorded judgment reasons.
    """
    records = []

    def log(*, run_id: str, problem: dict, result, note: str = "") -> None:
        if not isinstance(result, dict):
            result = result.to_dict(include_evidence=True)
        evidence = result.get("evidence") or []
        records.append({
            "test": request.node.nodeid,
            "ts": round(time.time(), 3),
            "run_id": run_id,
            "note": note,
            "problem": problem,
            "status": result.get("status"),
            "objective": result.get("objective"),
            "optimality_gap": result.get("optimality_gap"),
            "best_bound": result.get("best_bound"),
            "termination_reason": result.get("termination_reason"),
            "diagnostics": result.get("diagnostics"),
            "node_decisions": [
                {"node_id": e["node_id"], "decision": e["decision"], "reason": e["reason"]}
                for e in evidence
            ],
        })

    yield log

    if records:
        log_dir = Path(__file__).resolve().parent.parent / "test-logs"
        log_dir.mkdir(exist_ok=True)
        with (log_dir / "session.jsonl").open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
