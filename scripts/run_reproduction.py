#!/usr/bin/env python3
"""Run deterministic local reproduction experiments and save JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.contracts import SurvivalAnalysisRequest, SyntheticDatasetRequest
from app.service import run_analysis
from app.synthetic import generate_dataset
from app.tracing import utc_now_iso


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--sample-request", default="samples/analysis_request.json")
    parser.add_argument("--seed", type=int, default=311311)
    parser.add_argument("--sample-size", type=int, default=16)
    args = parser.parse_args()

    settings = Settings()
    output_dir = Path(args.output_dir)

    sample_path = Path(args.sample_request)
    hand_request = SurvivalAnalysisRequest.model_validate_json(sample_path.read_text())
    hand_result = run_analysis(
        hand_request,
        settings,
        analysis_id="reproduction-hand-fixture",
        request_id="reproduction-hand-request",
        created_at=utc_now_iso(),
    )
    _write_json(output_dir / "hand_fixture_result.json", hand_result)

    synthetic_request = SyntheticDatasetRequest(
        seed=args.seed,
        sample_size_per_group=args.sample_size,
        delayed_entry=True,
    )
    dataset = generate_dataset(synthetic_request)
    _write_json(output_dir / "synthetic_dataset.json", dataset)

    analysis_request = SurvivalAnalysisRequest(
        client_request_id="reproduction-synthetic",
        analysis_name="Deterministic synthetic equipment reproduction",
        confidence_level=0.95,
        significance_level=0.05,
        confidence_interval_method="log_log",
        groups=dataset["groups"],
        observations=dataset["observations"],
    )
    synthetic_result = run_analysis(
        analysis_request,
        settings,
        analysis_id="reproduction-synthetic",
        request_id="reproduction-synthetic-request",
        created_at=utc_now_iso(),
    )
    _write_json(output_dir / "synthetic_analysis_result.json", synthetic_result)

    summary = {
        "schema_version": settings.schema_version,
        "kernel_version": settings.kernel_version,
        "generator_version": settings.generator_version,
        "outputs": [
            "artifacts/hand_fixture_result.json",
            "artifacts/synthetic_dataset.json",
            "artifacts/synthetic_analysis_result.json",
        ],
        "medical_use": "not permitted; synthetic demonstration only",
    }
    _write_json(output_dir / "reproduction_manifest.json", summary)


if __name__ == "__main__":
    main()
