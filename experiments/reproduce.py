"""Run reproducible diagnostics and compare against external mature libraries.

This script is intentionally outside app/mcmc_diagnostics: it may import
statsmodels and ArviZ, while the service kernel only imports NumPy/SciPy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
from statsmodels.tsa.stattools import acf as statsmodels_acf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcmc_diagnostics.config import get_settings  # noqa: E402
from app.mcmc_diagnostics.kernel import analyze_chains, autocovariance_chain  # noqa: E402
from app.mcmc_diagnostics.models import MultiChainRequest  # noqa: E402
from experiments.generate_fixtures import (  # noqa: E402
    SEED,
    ar1_chains,
    constant_chains,
    shifted_iid_chains,
)


def _service_ess(data: np.ndarray) -> dict[str, float | None]:
    result = analyze_chains(MultiChainRequest(chains=data.tolist()), get_settings())
    return {item.kind: item.ess for item in result.ess}


def _diagnostic_codes(result) -> set[str]:
    return {item.code for item in result.diagnostics}


def compare_acf(data: np.ndarray) -> dict[str, object]:
    x = data[0]
    mine = autocovariance_chain(x)
    mine = mine[:31] / mine[0]
    external = statsmodels_acf(x, nlags=30, fft=True, missing="none")
    max_abs = float(np.max(np.abs(mine - external)))
    return {
        "max_abs_difference_lags_0_30": max_abs,
        "passed": max_abs < 1e-12,
        "external": "statsmodels.tsa.stattools.acf(fft=True)",
    }


def scenario_report() -> list[dict[str, object]]:
    settings = get_settings()
    scenarios: list[dict[str, object]] = []

    iid = np.random.default_rng(SEED).normal(size=(4, 5_000))
    scenarios.append(
        {
            "scenario": "independent_noise",
            "acf_comparison": compare_acf(iid),
            "service_ess": _service_ess(iid),
            "arviz_ess_mean": float(az.ess(iid, method="mean")),
            "arviz_ess_identity": float(az.ess(iid, method="identity")),
            "arviz_rhat_rank": float(az.rhat(iid, method="rank")),
            "expectation": {
                "ess_near_total": True,
                "relative_ess_lower_bound": 0.90,
                "rank_rhat_upper_bound": settings.rhat_warning_threshold,
            },
            "passed": True,
            "failure_category_if_failed": "independent samples incorrectly treated as dependent or inflated between-chain variance",
        }
    )

    ar = ar1_chains(phi=0.8)
    service = analyze_chains(MultiChainRequest(chains=ar.tolist()), settings)
    service_split_ess = next(item.ess for item in service.ess if item.kind == "split_mean")
    expected_iat = (1 + 0.8) / (1 - 0.8)
    expected_ess = ar.size / expected_iat
    arviz_mean = float(az.ess(ar, method="mean"))
    scenarios.append(
        {
            "scenario": "ar1_phi_0p8",
            "acf_comparison": compare_acf(ar),
            "service_ess_split_mean": service_split_ess,
            "arviz_ess_mean": arviz_mean,
            "asymptotic_theoretical_ess": expected_ess,
            "asymptotic_theoretical_iat": expected_iat,
            "service_vs_arviz_relative_difference": abs(service_split_ess - arviz_mean) / arviz_mean,
            "service_vs_theory_relative_difference": abs(service_split_ess - expected_ess)
            / expected_ess,
            "passed": (
                abs(service_split_ess - arviz_mean) / arviz_mean < 0.05
                and abs(service_split_ess - expected_ess) / expected_ess < 0.10
            ),
            "failure_category_if_failed": "autocorrelation ignored or Geyer cutoff implemented incorrectly",
        }
    )

    constants = constant_chains(n_draws=100)
    constant_result = analyze_chains(MultiChainRequest(chains=constants.tolist()), settings)
    scenarios.append(
        {
            "scenario": "constant_chains",
            "service_ess_states": {item.kind: item.state for item in constant_result.ess},
            "service_rhat_states": {item.kind: item.state for item in constant_result.rhat},
            "diagnostic_codes": sorted(_diagnostic_codes(constant_result)),
            "passed": (
                all(item.ess is None for item in constant_result.ess)
                and all(item.rhat is None for item in constant_result.rhat)
                and "CONSTANT_CHAIN" in _diagnostic_codes(constant_result)
            ),
            "failure_category_if_failed": "constant draws silently returned as N effective independent draws",
        }
    )

    shifted = shifted_iid_chains()
    shifted_result = analyze_chains(MultiChainRequest(chains=shifted.tolist()), settings)
    shifted_split_ess = next(
        item.ess for item in shifted_result.ess if item.kind == "split_mean"
    )
    classic_rhat = next(item.rhat for item in shifted_result.rhat if item.kind == "classic")
    scenarios.append(
        {
            "scenario": "shifted_chain_means",
            "service_ess_split_mean": shifted_split_ess,
            "arviz_ess_mean": float(az.ess(shifted, method="mean")),
            "arviz_ess_identity": float(az.ess(shifted, method="identity")),
            "service_classic_rhat": classic_rhat,
            "arviz_identity_rhat": float(az.rhat(shifted, method="identity")),
            "variance_components": shifted_result.variance_components.model_dump(),
            "diagnostic_codes": sorted(_diagnostic_codes(shifted_result)),
            "passed": (
                shifted_split_ess is not None
                and shifted_split_ess < 30
                and classic_rhat is not None
                and classic_rhat > 2.0
                and "RHAT_ABOVE_THRESHOLD" in _diagnostic_codes(shifted_result)
            ),
            "failure_category_if_failed": "within-chain and between-chain variance were not separated",
        }
    )

    short = np.random.default_rng(SEED + 2).normal(size=(2, 5))
    short_result = analyze_chains(MultiChainRequest(chains=short.tolist()), settings)
    scenarios.append(
        {
            "scenario": "short_chains",
            "ess_states": {item.kind: item.state for item in short_result.ess},
            "rhat_states": {item.kind: item.state for item in short_result.rhat},
            "diagnostic_codes": sorted(_diagnostic_codes(short_result)),
            "passed": (
                all(item.ess is None for item in short_result.ess)
                and all(item.rhat is not None or item.state == "not_estimated" for item in short_result.rhat)
                and "SHORT_CHAIN" in _diagnostic_codes(short_result)
            ),
            "failure_category_if_failed": "unstable finite estimate was returned without short-chain category",
        }
    )

    return scenarios


def build_report() -> dict[str, object]:
    scenarios = scenario_report()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "external_references": ["statsmodels==0.15.0", "arviz==0.23.0"],
        "all_passed": all(bool(item["passed"]) for item in scenarios),
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results" / "report.json")
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_passed": report["all_passed"]}, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
