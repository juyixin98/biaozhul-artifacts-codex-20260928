"""Deterministic local reproduction experiments.

Run with:
    python -m mcmc_diagnostics.experiments --output experiments/report.json

The test oracle used here is independent of ``kernel.py``: fixtures come from
explicit NumPy equations and reference statistics come from ArviZ.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .kernel import analyze_chains


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    expected_failure_category: str | None
    expected_property: str
    chains: list[list[float]]


def make_cases(seed: int = 315) -> list[ExperimentCase]:
    rng = np.random.default_rng(seed)
    chains = 4
    draws = 2_000

    iid = rng.normal(size=(chains, draws))

    phi = 0.8
    innovation_scale = np.sqrt(1.0 - phi**2)
    ar1 = np.zeros((chains, draws))
    for chain in range(chains):
        innovations = rng.normal(scale=innovation_scale, size=draws)
        ar1[chain, 0] = innovations[0]
        for draw in range(1, draws):
            ar1[chain, draw] = phi * ar1[chain, draw - 1] + innovations[draw]

    shifted = np.vstack(
        (
            rng.normal(loc=-3.0, scale=1.0, size=(2, 1_000)),
            rng.normal(loc=3.0, scale=1.0, size=(2, 1_000)),
        )
    )
    constant = np.full((2, 1_000), 7.0)
    short = rng.normal(size=(2, 3))

    return [
        ExperimentCase(
            "independent_noise",
            None,
            "ESS should be near nominal count; R-hat should be near 1.",
            iid.tolist(),
        ),
        ExperimentCase(
            "ar1_phi_0.8",
            None,
            "ESS should degrade to roughly (1-phi)/(1+phi)*MN, about 0.111*MN.",
            ar1.tolist(),
        ),
        ExperimentCase(
            "offset_chain_means",
            "RHAT_ABOVE_THRESHOLD",
            "Separated chain means should produce low ESS and R-hat above 1.01.",
            shifted.tolist(),
        ),
        ExperimentCase(
            "constant_chains",
            "CONSTANT_CHAIN",
            "Constant chains must be separately reported as non-informative.",
            constant.tolist(),
        ),
        ExperimentCase(
            "short_chains",
            "SHORT_CHAIN",
            "Fewer than four draws must suppress split and Geyer estimates.",
            short.tolist(),
        ),
    ]


def _arviz_reference(matrix: np.ndarray) -> dict[str, Any]:
    try:
        import arviz as az
    except ImportError:
        return {"available": False}
    reference: dict[str, Any] = {
        "available": True,
        "ess_mean": _finite_float(az.ess(matrix, method="mean")),
        "ess_bulk": _finite_float(az.ess(matrix, method="bulk")),
        "ess_identity": _finite_float(az.ess(matrix, method="identity")),
    }
    if matrix.shape[0] > 1:
        reference["rhat_rank"] = _finite_float(az.rhat(matrix, method="rank"))
    else:
        reference["rhat_rank"] = None
    return reference


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def run_case(case: ExperimentCase) -> dict[str, Any]:
    matrix = np.asarray(case.chains, dtype=float)
    result = analyze_chains(
        matrix,
        default_max_acf_lag=10,
        rhat_warning_threshold=1.01,
        ess_low_fraction=0.10,
    )
    finding_codes = [item["code"] for item in result["findings"]]
    reference = _arviz_reference(matrix)

    total = matrix.size
    ess = result["effective_sample_size"]
    checks: dict[str, Any] = {}
    if case.name == "independent_noise":
        checks["ess_near_n"] = (
            _finite_float(ess["identity_original_chains"]["value"]) is not None
            and 0.75 * total
            <= ess["identity_original_chains"]["value"]
            <= 1.25 * total
        )
        checks["matches_arviz_ess"] = (
            reference.get("available")
            and abs(
                ess["mean_split_chains"]["value"] - reference["ess_mean"]
            )
            < 1e-8
        )
    elif case.name == "ar1_phi_0.8":
        theoretical_total = total * (1.0 - 0.8) / (1.0 + 0.8)
        observed = ess["identity_original_chains"]["value"]
        checks["theoretical_degradation_ratio"] = (
            observed / total if observed is not None else None
        )
        checks["within_25_percent_of_ar1_theory"] = (
            observed is not None
            and 0.75 * theoretical_total <= observed <= 1.25 * theoretical_total
        )
        checks["matches_arviz_ess"] = (
            reference.get("available")
            and abs(
                ess["bulk_split_rank_normalized"]["value"] - reference["ess_bulk"]
            )
            < 1e-8
        )
    elif case.name == "offset_chain_means":
        checks["expected_failure_category_present"] = (
            case.expected_failure_category in finding_codes
        )
        checks["low_identity_ess"] = (
            ess["identity_original_chains"]["value"] is not None
            and ess["identity_original_chains"]["value"] < 0.05 * total
        )
        checks["rhat_above_1_01"] = (
            result["convergence"]["rank_split_chains"]["value"] > 1.01
        )
        checks["matches_arviz_rhat"] = (
            reference.get("available")
            and abs(
                result["convergence"]["rank_split_chains"]["value"]
                - reference["rhat_rank"]
            )
            < 1e-10
        )
    elif case.name == "constant_chains":
        checks["expected_failure_category_present"] = (
            case.expected_failure_category in finding_codes
        )
        checks["ess_not_estimated"] = (
            ess["identity_original_chains"]["status"]
            == "not_estimated_constant"
        )
        checks["rhat_not_estimated"] = (
            result["convergence"]["rank_split_chains"]["status"]
            == "not_estimated_constant"
        )
        checks["arviz_constant_convention_recorded"] = (
            reference.get("available") and reference["ess_mean"] == total
        )
    elif case.name == "short_chains":
        checks["expected_failure_category_present"] = (
            case.expected_failure_category in finding_codes
        )
        checks["split_ess_suppressed"] = (
            ess["mean_split_chains"]["status"] == "not_estimated_short_chain"
        )
        checks["rank_rhat_suppressed"] = (
            result["convergence"]["rank_split_chains"]["status"]
            == "not_estimated_short_chain"
        )

    return {
        "name": case.name,
        "expected_property": case.expected_property,
        "expected_failure_category": case.expected_failure_category,
        "observed_status": result["status"],
        "observed_finding_codes": finding_codes,
        "observed": {
            "ess_identity": ess["identity_original_chains"],
            "ess_mean_split": ess["mean_split_chains"],
            "ess_bulk_split_rank": ess["bulk_split_rank_normalized"],
            "rank_split_rhat": result["convergence"]["rank_split_chains"],
        },
        "independent_reference": reference,
        "checks": checks,
        "all_checks_passed": all(
            value is True
            for key, value in checks.items()
            if key.startswith(("expected", "ess_", "rhat_", "low_", "within", "matches"))
        ),
    }


def run_experiments(seed: int = 315) -> dict[str, Any]:
    cases = make_cases(seed=seed)
    reports = [run_case(case) for case in cases]
    return {
        "schema_version": "mcmc-reproduction/v1",
        "seed": seed,
        "generator": "Local NumPy equations; fixtures are not generated by the tested kernel.",
        "case_count": len(reports),
        "all_checks_passed": all(item["all_checks_passed"] for item in reports),
        "cases": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/report.json"),
        help="JSON report path.",
    )
    args = parser.parse_args()
    report = run_experiments(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict_shim(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} with {report['case_count']} cases; "
        f"passed={report['all_checks_passed']}"
    )


def asdict_shim(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): asdict_shim(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [asdict_shim(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
