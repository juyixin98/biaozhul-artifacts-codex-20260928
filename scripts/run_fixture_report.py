#!/usr/bin/env python3
"""Run the bundled fixture through the mutation engine and print a compact report."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mutation_backend.engine import run_mutation_analysis
from mutation_backend.models import MutationOperator, MutationRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-root",
        default=str(Path(tempfile.gettempdir()) / "python-mutation-backend-bundled-run"),
    )
    parser.add_argument("--database", default="data/mutation-evidence.sqlite3")
    args = parser.parse_args()

    fixture = ROOT / "fixtures" / "sample_algorithm_pkg"
    request = MutationRequest(
        request_id="bundled-fixture",
        project_path=fixture,
        test_command=["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        source_paths=["algorithm_pkg/**/*.py"],
        operators=[
            MutationOperator.COMPARISON,
            MutationOperator.ARITHMETIC,
            MutationOperator.CONDITIONAL_INVERT,
            MutationOperator.CONDITION_ERASE,
        ],
        baseline_timeout_seconds=10,
        mutant_timeout_seconds=0.5,
    )
    report = run_mutation_analysis(
        request,
        work_root=Path(args.work_root),
        request_id=request.request_id,
    )

    from mutation_backend.storage import EvidenceStore

    Path(args.database).parent.mkdir(parents=True, exist_ok=True)
    EvidenceStore(Path(args.database)).save_report(report)
    payload = report.model_dump()
    compact = {
        "request_id": payload["request_id"],
        "status": payload["status"],
        "engine_version": payload["engine_version"],
        "timeouts": payload["timeouts"],
        "summary": payload["summary"],
        "source_unchanged": payload["source_unchanged"],
        "failures": [
            {
                "mutant_id": item["mutant_id"],
                "fate": item["fate"],
                "failure_category": item["failure_category"],
                "path": item["location"]["path"],
                "line": item["location"]["line"],
                "before": item["before"],
                "after": item["after"],
                "reason": item["reason"],
            }
            for item in payload["failures"]
        ],
        "uncertainties": [
            {
                "mutant_id": item["mutant_id"],
                "fate": item["fate"],
                "basis": item["equivalent_basis"],
            }
            for item in payload["uncertainties"]
        ],
        "database": args.database,
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
