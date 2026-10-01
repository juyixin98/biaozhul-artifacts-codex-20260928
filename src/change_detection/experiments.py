from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import SCHEMA_VERSION, __version__
from .dp import solve
from .errors import ChangeDetectionError
from .reference import enumerate_best_segmentation
from .service import canonical_input_hash, utc_now_iso


@dataclass(frozen=True)
class Scenario:
    name: str
    values: list[float] | list[int]
    min_segment_length: int
    penalty: float
    expected_breakpoints: tuple[int, ...]
    expected_segment_count: int
    expected_status: str
    decision: str
    reference_enabled: bool


def build_scenarios() -> list[Scenario]:
    rng = np.random.default_rng(20261001)
    noisy_step = np.concatenate(
        [
            scipy_stats.norm.rvs(loc=0.0, scale=0.05, size=8, random_state=rng),
            scipy_stats.norm.rvs(loc=5.0, scale=0.05, size=8, random_state=rng),
        ]
    ).tolist()
    return [
        Scenario(
            name="integer_mean_step_zero_penalty",
            values=[0, 0, 0, 10, 10, 10],
            min_segment_length=1,
            penalty=0.0,
            expected_breakpoints=(3,),
            expected_segment_count=2,
            expected_status="PASS",
            decision="zero penalty should split a clean step without singleton fragmentation",
            reference_enabled=True,
        ),
        Scenario(
            name="single_outlier_moderate_penalty",
            values=[0, 0, 0, 10, 0, 0, 0],
            min_segment_length=1,
            penalty=20.0,
            expected_breakpoints=(3, 4),
            expected_segment_count=3,
            expected_status="PASS",
            decision="isolating a large spike is cheaper once SSE exceeds added penalties",
            reference_enabled=True,
        ),
        Scenario(
            name="single_outlier_huge_penalty",
            values=[0, 0, 0, 10, 0, 0, 0],
            min_segment_length=1,
            penalty=200.0,
            expected_breakpoints=(),
            expected_segment_count=1,
            expected_status="PASS",
            decision="very large per-segment penalty must select one segment",
            reference_enabled=True,
        ),
        Scenario(
            name="minimum_segment_length_two",
            values=[0, 0, 10, 10],
            min_segment_length=2,
            penalty=0.0,
            expected_breakpoints=(2,),
            expected_segment_count=2,
            expected_status="PASS",
            decision="only boundaries that leave legal suffixes are considered",
            reference_enabled=True,
        ),
        Scenario(
            name="noisy_gaussian_step_reproduction",
            values=[round(float(value), 8) for value in noisy_step],
            min_segment_length=2,
            penalty=0.5,
            expected_breakpoints=(8,),
            expected_segment_count=2,
            expected_status="PASS",
            decision="seeded Gaussian noise should still expose the planted level shift",
            reference_enabled=False,
        ),
        Scenario(
            name="pruned_and_unpruned_must_agree",
            values=[0, 10, 0, 10, 0, 10, 0, 10, 0, 10],
            min_segment_length=1,
            penalty=50.0,
            expected_breakpoints=(),
            expected_segment_count=1,
            expected_status="PASS",
            decision="pruning cannot alter objective or deterministic tie choice",
            reference_enabled=True,
        ),
    ]


def _reference_matches(result: Any, scenario: Scenario) -> dict[str, Any]:
    reference = enumerate_best_segmentation(
        scenario.values,
        scenario.min_segment_length,
        scenario.penalty,
    )
    if reference.exact:
        dp_objective = Fraction(str(result.objective))
        objective_matches = dp_objective == reference.objective
        objective_delta = float(abs(dp_objective - reference.objective))
    else:
        objective_delta = abs(float(result.objective) - float(reference.objective))
        scale = max(1.0, abs(float(reference.objective)))
        objective_matches = objective_delta <= 1e-7 * scale

    breakpoint_matches = tuple(result.breakpoints) == tuple(reference.breakpoints)
    return {
        "reference_considered_segmentations": reference.considered_segmentations,
        "reference_objective": str(reference.objective),
        "reference_breakpoints": list(reference.breakpoints),
        "reference_exact": reference.exact,
        "reference_objective_delta": objective_delta,
        "reference_judgement": (
            "PASS" if objective_matches and breakpoint_matches else "FAIL"
        ),
    }


def run_scenarios(output_dir: str = "experiments/latest", run_id: str | None = None) -> pd.DataFrame:
    run_id = run_id or f"local-{utc_now_iso().replace(':', '')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, scenario in enumerate(build_scenarios(), start=1):
        request = {
            "values": scenario.values,
            "min_segment_length": scenario.min_segment_length,
            "penalty": scenario.penalty,
            "prune": True,
            "verify_pruning": True,
        }
        events: list[tuple[str, dict]] = []
        row: dict[str, Any] = {
            "run_id": run_id,
            "scenario_index": index,
            "scenario": scenario.name,
            "n": len(scenario.values),
            "min_segment_length": scenario.min_segment_length,
            "penalty": scenario.penalty,
            "input_sha256": canonical_input_hash(request),
            "core_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "expected_breakpoints": list(scenario.expected_breakpoints),
            "expected_segment_count": scenario.expected_segment_count,
            "decision_basis": scenario.decision,
        }
        try:
            result = solve(
                scenario.values,
                min_segment_length=scenario.min_segment_length,
                penalty=scenario.penalty,
                prune=True,
                verify_pruning=True,
                progress=lambda event, payload: events.append((event, payload)),
                run_id=run_id,
            )
            unpruned = solve(
                scenario.values,
                min_segment_length=scenario.min_segment_length,
                penalty=scenario.penalty,
                prune=False,
                verify_pruning=False,
                progress=None,
                run_id=run_id,
            )
            row.update(
                {
                    "status": "PASS",
                    "objective": result.objective,
                    "sse": result.sse,
                    "total_penalty": result.total_penalty,
                    "actual_breakpoints": list(result.breakpoints),
                    "actual_segment_count": result.segment_count,
                    "transition_count": result.transition_count,
                    "pruned_transitions": result.pruned_transitions,
                    "unpruned_transition_count": unpruned.transition_count,
                    "verification_judgement": (
                        "PASS"
                        if result.breakpoints == unpruned.breakpoints
                        and abs(result.objective - unpruned.objective)
                        <= 1e-9 * max(1.0, abs(result.objective), abs(unpruned.objective))
                        else "FAIL"
                    ),
                    "event_count": len(events),
                    "error_category": "",
                    "error_message": "",
                }
            )
            if result.breakpoints != scenario.expected_breakpoints:
                row["status"] = "FAIL"
            if result.segment_count != scenario.expected_segment_count:
                row["status"] = "FAIL"
            if scenario.reference_enabled:
                row.update(_reference_matches(result, scenario))
                if row["reference_judgement"] == "FAIL":
                    row["status"] = "FAIL"
            else:
                row["reference_judgement"] = "NOT_RUN"
        except ChangeDetectionError as exc:
            row.update(
                {
                    "status": "FAIL",
                    "error_category": exc.category,
                    "error_message": exc.message,
                    "verification_judgement": "NOT_RUN",
                    "reference_judgement": "NOT_RUN",
                }
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "results.csv", index=False)
    summary = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "core_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "total": len(frame),
        "passed": int((frame["status"] == "PASS").sum()),
        "failed": int((frame["status"] == "FAIL").sum()),
        "scenarios": frame[["scenario", "status", "objective", "actual_breakpoints"]].to_dict(
                    orient="records"
                ),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return frame
