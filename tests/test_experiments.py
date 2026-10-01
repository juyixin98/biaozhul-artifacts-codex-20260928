from __future__ import annotations

import json
from pathlib import Path

from change_detection.experiments import build_scenarios, run_scenarios


def test_reproduction_scenarios_have_independent_judgement_and_artifacts(tmp_path: Path) -> None:
    frame = run_scenarios(output_dir=str(tmp_path), run_id="deterministic-test-run")

    assert len(frame) == len(build_scenarios())
    assert (frame["status"] == "PASS").all()
    assert frame["run_id"].eq("deterministic-test-run").all()
    assert (frame["verification_judgement"] == "PASS").all()
    assert frame.loc[frame["reference_judgement"] == "PASS", "reference_considered_segmentations"].gt(0).all()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] == 0
    assert summary["passed"] == len(frame)
    assert summary["core_version"] == "1.0.0"
    assert (tmp_path / "results.csv").exists()
