"""Structured per-run replay logs for the independent test suite.

Every test that makes a correctness claim creates one JSONL record under
``test_runs/<run_id>.jsonl`` containing a stable run id, the decision under
test, concrete expected/observed intermediate values and the reason the
assertion holds. Failures include the exact failure class. The logs are
kept in the repository workspace (not a throwaway tmp dir) so a rejected
run can be replayed from its seed/inputs and audited afterwards.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

RUN_LOG_DIR = Path(__file__).resolve().parent.parent / "test_runs"


class RunLogger:
    def __init__(self, test_name: str, run_id: str | None = None) -> None:
        self.test_name = test_name
        # The run id is deterministic by test name so a failure can be
        # replayed to the same log path; callers may still force a suffix.
        self.run_id = run_id or test_name
        RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RUN_LOG_DIR / f"{self.run_id}.jsonl"
        # Fresh path per test execution: append mode keeps the record sequence.
        self.path.write_text("", encoding="utf-8")

    def record(
        self,
        decision: str,
        reason: str,
        expected: Any = None,
        observed: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        record = {
            "run_id": self.run_id,
            "test": self.test_name,
            "decision": decision,
            "reason": reason,
            "expected": expected,
            "observed": observed,
        }
        record.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + os.linesep)
        return record

    def read(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
