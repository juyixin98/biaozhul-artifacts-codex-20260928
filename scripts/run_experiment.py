#!/usr/bin/env python3
"""Run deterministic local calibration experiments and save versioned results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.contracts import RunRequest  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.repository import RunRepository  # noqa: E402
from app.service import CalibrationService  # noqa: E402


def sample(record_id: str, y: int, p: float, w: float = 1.0) -> dict:
    return {"id": record_id, "y": y, "p": p, "w": w}


def perfect_scenario() -> dict:
    fit = [
        sample("perfect-fit-0", 0, 0.0),
        sample("perfect-fit-1", 0, 0.1),
        sample("perfect-fit-2", 0, 0.1),
        sample("perfect-fit-3", 1, 0.9),
        sample("perfect-fit-4", 1, 1.0),
    ]
    evaluation = [
        sample("perfect-eval-0", 0, 0.0),
        sample("perfect-eval-1", 0, 0.1),
        sample("perfect-eval-2", 1, 0.9),
        sample("perfect-eval-3", 1, 1.0),
    ]
    return {"fit": fit, "eval": evaluation}


def overconfident_scenario() -> dict:
    fit = [
        sample("over-fit-0", 0, 0.9),
        sample("over-fit-1", 1, 0.9),
        sample("over-fit-2", 0, 0.1),
        sample("over-fit-3", 1, 0.1),
    ]
    evaluation = [
        sample("over-eval-0", 0, 0.9),
        sample("over-eval-1", 0, 0.9),
        sample("over-eval-2", 1, 0.9),
        sample("over-eval-3", 1, 0.9),
        sample("over-eval-4", 0, 0.1),
        sample("over-eval-5", 0, 0.1),
        sample("over-eval-6", 1, 0.1),
        sample("over-eval-7", 1, 0.1),
    ]
    return {"fit": fit, "eval": evaluation}


def constant_scenario() -> dict:
    fit = [sample(f"const-fit-{i}", i % 2, 0.5) for i in range(4)]
    evaluation = [sample(f"const-eval-{i}", i % 2, 0.5) for i in range(4)]
    return {"fit": fit, "eval": evaluation}


def extreme_weight_scenario() -> dict:
    fit = [
        sample("weight-fit-0", 0, 0.2),
        sample("weight-fit-1", 1, 0.8),
    ]
    evaluation = [
        sample("weight-eval-0", 0, 0.0),
        sample("weight-eval-1", 1, 0.0, 999.0),
    ]
    return {"fit": fit, "eval": evaluation}


SCENARIOS = {
    "perfect": perfect_scenario,
    "overconfident": overconfident_scenario,
    "constant": constant_scenario,
    "extreme_weights": extreme_weight_scenario,
}


def build_request(scenario: str, calibrator: str, run_id: str) -> dict:
    payload = SCENARIOS[scenario]()
    return {
        "run_id": run_id,
        "fit": payload["fit"],
        "eval": payload["eval"],
        "calibrator": {"type": calibrator},
        "fixed_bins": {"edges": [0.0, 0.5, 1.0]},
        "quantile_bins": {"n_bins": 3},
        "metadata": {"scenario": scenario, "local_synthetic": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/experiment_runs.sqlite3")
    parser.add_argument("--output", default="outputs/experiment_results.json")
    parser.add_argument("--log-file", default="logs/experiment.log")
    args = parser.parse_args()

    configure_logging("INFO", args.log_file)
    settings = Settings(database_path=Path(args.db), log_file=Path(args.log_file))
    service = CalibrationService(settings, RunRepository(settings.database_path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in ("perfect", "overconfident", "constant", "extreme_weights"):
        run_id = f"{scenario}-isotonic"
        request = RunRequest.model_validate(build_request(scenario, "isotonic", run_id))
        result = service.create_run(request)
        results.append(json.loads(result.model_dump_json()))

    output_path.write_text(json.dumps({"runs": results}, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} deterministic runs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
