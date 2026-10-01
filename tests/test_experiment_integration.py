"""Integration test for the reproducible local experiment entry point."""

from __future__ import annotations

import json
import subprocess
import sys


def test_experiment_script_writes_versioned_results(tmp_path, run_identity):
    db_path = tmp_path / "experiments.sqlite3"
    output_path = tmp_path / "results.json"
    log_path = tmp_path / "experiment.log"
    run_identity.emit(
        "subprocess_experiment",
        db=str(db_path),
        output=str(output_path),
        judgment="process must exit zero and write four SUCCEEDED runs",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--db",
            str(db_path),
            "--output",
            str(output_path),
            "--log-file",
            str(log_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "wrote 4 deterministic runs" in completed.stdout

    payload = json.loads(output_path.read_text())
    assert len(payload["runs"]) == 4
    for run in payload["runs"]:
        assert run["status"] == "SUCCEEDED"
        assert run["input_hash"]
        assert run["versions"]["kernel_version"]
        assert run["calibrator"]["disjoint_fit_and_evaluation"] is True
        for probability_set in ("raw", "calibrated"):
            for scheme in ("fixed", "equal_frequency"):
                stats = run["analyses"][probability_set][scheme]
                assert stats["weight_conservation_check"]["passed"] is True
                assert stats["decomposition_check"]["passed"] is True

    # Re-running deterministic IDs must surface a conflict rather than overwrite.
    repeated = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--db",
            str(db_path),
            "--output",
            str(output_path),
            "--log-file",
            str(log_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "run_id_conflict" in repeated.stderr
