from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from survey_backend.statistical.data import prepare_records
from survey_backend.synthesis.generator import SyntheticSurvey, TrueParameters, generate_survey
from survey_backend.statistical.analysis import run_comparison


TRUE = TrueParameters()
TARGET_BETA = np.asarray([TRUE.beta_q1, TRUE.beta_q2, TRUE.intercept])


def _baseline_ols(survey: SyntheticSurvey) -> tuple[np.ndarray, list[dict[str, float]]]:
    rows = np.flatnonzero(survey.eligible)
    x = np.column_stack([survey.q1[rows], survey.q2[rows], np.ones(rows.size)])
    beta, _, _, _ = np.linalg.lstsq(x, survey.y[rows], rcond=None)
    residual = survey.y[rows] - x @ beta
    sigma2 = float(residual @ residual / (rows.size - 3))
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    intervals = [
        {
            "lower": float(beta[i] - 1.96 * se[i]),
            "upper": float(beta[i] + 1.96 * se[i]),
            "estimate": float(beta[i]),
        }
        for i in range(3)
    ]
    return beta, intervals


def _method_payload(response, method_name: str) -> dict[str, object]:
    return next(result for result in response.results if result.method == method_name)


def run_experiment(repetitions: int, n: int, seed: int, m: int, burn_in: int) -> dict[str, object]:
    mechanisms = ("mcar", "mar", "mnar")
    estimates: dict[str, dict[str, list[float]]] = {
        mechanism: {
            "baseline": [],
            "complete_case": [],
            "mean_imputation": [],
            "multiple_imputation": [],
        }
        for mechanism in mechanisms
    }
    coverage: dict[str, dict[str, list[int]]] = {
        mechanism: {
            "baseline": [],
            "complete_case": [],
            "mean_imputation": [],
            "multiple_imputation": [],
        }
        for mechanism in mechanisms
    }

    for repeat in range(repetitions):
        run_seed = seed + repeat * 1009
        for mechanism in mechanisms:
            survey = generate_survey(
                n=n,
                seed=run_seed,
                mechanism=mechanism,
                missing_probability=0.22 if mechanism != "mcar" else 0.18,
                ineligible_probability=0.10,
            )
            baseline_beta, baseline_intervals = _baseline_ols(survey)
            estimates[mechanism]["baseline"].append(baseline_beta.tolist())
            coverage[mechanism]["baseline"].append(
                [
                    interval["lower"] <= TARGET_BETA[i] <= interval["upper"]
                    for i, interval in enumerate(baseline_intervals)
                ]
            )

            response = run_comparison(
                prepare_records(survey.to_records()),
                analysis_id=f"experiment-{mechanism}-{repeat}",
                version=1,
                dataset_id=f"synthetic-{mechanism}",
                request_id=f"experiment-{mechanism}-{repeat}",
                request_hash=f"seed-{run_seed}",
                confidence=0.95,
                seed=run_seed,
                imputations=m,
                burn_in=burn_in,
            )
            for method in (
                "complete_case",
                "mean_imputation",
                "multiple_imputation",
            ):
                payload = _method_payload(response, method)
                beta = np.asarray([coefficient.estimate for coefficient in payload.coefficients])
                estimates[mechanism][method].append(beta.tolist())
                coverage[mechanism][method].append(
                    [
                        coefficient.ci_lower <= TARGET_BETA[i] <= coefficient.ci_upper
                        for i, coefficient in enumerate(payload.coefficients)
                    ]
                )

    summary: dict[str, object] = {
        "config": {
            "repetitions": repetitions,
            "n_per_dataset": n,
            "seed": seed,
            "m": m,
            "burn_in": burn_in,
            "true_parameters": asdict(TRUE),
            "target_coefficient_order": ["beta_q1", "beta_q2", "intercept"],
        },
        "mechanisms": {},
    }
    for mechanism in mechanisms:
        method_summary = {}
        for method, matrix in estimates[mechanism].items():
            values = np.asarray(matrix)
            covers = np.asarray(coverage[mechanism][method])
            method_summary[method] = {
                "mean_estimate": np.mean(values, axis=0).tolist(),
                "bias": (np.mean(values, axis=0) - TARGET_BETA).tolist(),
                "rmse": np.sqrt(np.mean((values - TARGET_BETA) ** 2, axis=0)).tolist(),
                "coverage_95": np.mean(covers, axis=0).tolist(),
            }
        summary["mechanisms"][mechanism] = method_summary
    return summary


def write_markdown(summary: dict[str, object], target: Path) -> None:
    lines = [
        "# Reproduction Results",
        "",
        "Bias and 95% interval coverage are computed only on local synthetic data.",
        "",
        "| Mechanism | Method | q1 bias | q2 bias | intercept bias | q1 coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    mechanisms = summary["mechanisms"]
    for mechanism, methods in mechanisms.items():
        for method, values in methods.items():
            lines.append(
                f"| {mechanism} | {method} | "
                f"{values['bias'][0]:.4f} | {values['bias'][1]:.4f} | "
                f"{values['bias'][2]:.4f} | {values['coverage_95'][0]:.3f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: complete cases assume MCAR; mean imputation is a deterministic benchmark;",
            "multiple imputation is approximately calibrated under the implemented MAR model but is not",
            "guaranteed unbiased when missingness is MNAR.",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--n", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--burn-in", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/reproduction_results.json"))
    args = parser.parse_args()
    summary = run_experiment(args.repetitions, args.n, args.seed, args.m, args.burn_in)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output.with_suffix(".md"))
    print(args.output)


if __name__ == "__main__":
    main()
