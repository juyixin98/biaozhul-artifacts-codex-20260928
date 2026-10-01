"""Shared pytest helpers."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from core.logging_utils import JsonLineLogger


def make_case_logger(log_dir: Path, test_name: str) -> tuple[JsonLineLogger, Path]:
    run_id = f"pytest-{test_name}-{uuid.uuid4().hex[:8]}"
    path = log_dir / f"{run_id}.jsonl"
    logger = JsonLineLogger(str(path))

    def log_case(input_digest: str, expected: object, actual: object, passed: bool, basis: str) -> dict:
        return logger.log(
            "assertion",
            run_id,
            input_digest=input_digest,
            expected=expected,
            actual=actual,
            passed=passed,
            judgment_basis=basis,
            status="success" if passed else "failed",
        )

    logger.run_id = run_id  # type: ignore[attr-defined]
    logger.log_case = log_case  # type: ignore[attr-defined]
    return logger, path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
