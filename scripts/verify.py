#!/usr/bin/env python3
"""End-to-end verification for the mutation testing backend.

Runs the engine twice against the bundled fixture package and checks:
  1. known mutant fates (hand-written reference, see tests/)
  2. original source tree digest is unchanged
  3. repeated runs produce identical fates
Exits non-zero on any failed check.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.models import MutationOperator, RunRequest, RunStatus
from app.runner import MutationRunner, source_digest
from app.store import EvidenceStore

FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sample_pkg"
)

EXPECTED_COUNTS = {
    "total": 22, "killed": 16, "survived": 2, "undetermined": 4,
    "timeout": 0, "invalid_syntax": 0, "startup_failure": 0,
}


def main() -> int:
    failures = []
    work = Path(tempfile.mkdtemp(prefix="mut-verify-"))
    digest_before = source_digest(FIXTURE)
    fates = []
    summaries = []
    for i in range(2):
        db = work / f"run{i}.db"
        settings = Settings(db_path=str(db))
        store = EvidenceStore(settings.db_path)
        runner = MutationRunner(settings, store)
        req = RunRequest(
            source_dir=str(FIXTURE), test_dir="tests",
            timeout_seconds=15.0, requester="verify-script",
            operators=list(MutationOperator),
        )
        summary = runner.run(req)
        summaries.append(summary)
        fates.append(
            {(m.original_segment, m.mutated_segment): m.outcome.value
             for m in store.list_mutants(summary.run_id)}
        )

    s = summaries[0]
    if s.status is not RunStatus.COMPLETED:
        failures.append(f"run status {s.status}, expected completed")
    for key, expected in EXPECTED_COUNTS.items():
        actual = getattr(s, key)
        if actual != expected:
            failures.append(f"{key}: got {actual}, expected {expected}")
    if source_digest(FIXTURE) != digest_before:
        failures.append("original source tree changed during runs")
    if fates[0] != fates[1]:
        failures.append("repeated runs produced different fates")

    print(f"run_id={s.run_id} status={s.status.value} "
          f"total={s.total} killed={s.killed} survived={s.survived} "
          f"undetermined={s.undetermined}")
    print(f"source digest stable: "
          f"{source_digest(FIXTURE) == digest_before}")
    print(f"repeatable: {fates[0] == fates[1]}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
