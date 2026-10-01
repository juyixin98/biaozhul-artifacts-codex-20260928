"""Reproducible comparison experiment over known missingness mechanisms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import MissingnessMechanism
from .kernel import build_design, run_method
from .repository import SQLiteResultRepository
from .synthetic import generate_survey


@dataclass(frozen=True)
class ExperimentRow:
    request_id: str
    seed: int
    n_records: int
    mechanism: str
    method: str
    model_version: str
    estimate: float
    variance: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_missing: int
    imputation_count: int | None
    target_value: float | None
    bias: float | None


def _records_to_arrays(records):
    y = np.asarray(
        [np.nan if record.rating is None else record.rating for record in records], dtype=float
    )
    x = np.asarray(
        [
            [record.age_z, record.member, record.used_support, record.contact_frequency]
            for record in records
        ],
        dtype=float,
    )
    return y, x


def run_comparison(
    repository: SQLiteResultRepository | None = None,
    n: int = 1000,
    missing_probability: float = 0.20,
    m: int = 30,
    base_seed: int = 20260930,
) -> list[ExperimentRow]:
    mechanisms = [
        MissingnessMechanism.complete,
        MissingnessMechanism.mcar,
        MissingnessMechanism.mar,
        MissingnessMechanism.mnar,
    ]
    rows: list[ExperimentRow] = []

    for method_seed_offset, mechanism in enumerate(mechanisms):
        seed = base_seed
        generated = generate_survey(
            n=n,
            mechanism=mechanism,
            missing_probability=missing_probability,
            seed=seed,
            missing_seed=base_seed + 91_000,
            request_id=f"experiment-{mechanism.value}-{seed}",
        )
        y, x = _records_to_arrays(generated.records)
        design = build_design(
            [record.record_id for record in generated.records], y, x, np.isnan(y)
        )
        methods = ["complete_data"] if mechanism == MissingnessMechanism.complete else [
            "complete_case",
            "mean_imputation",
            "multiple_imputation",
        ]
        for method in methods:
            result = run_method(
                design, method=method, m=m, seed=seed + 77_000 + method_seed_offset
            )
            row = ExperimentRow(
                request_id=f"experiment-{mechanism.value}-{seed}",
                seed=seed,
                n_records=n,
                mechanism=mechanism.value,
                method=result.method,
                model_version=result.model_version,
                estimate=result.estimate,
                variance=result.variance,
                standard_error=result.standard_error,
                ci_low=result.ci95_low,
                ci_high=result.ci95_high,
                n_missing=result.n_missing,
                imputation_count=result.imputation_count,
                target_value=generated.true_mean_rating,
                bias=result.estimate - generated.true_mean_rating,
            )
            rows.append(row)
            if repository is not None:
                repository.save_experiment_row(asdict(row))
    return rows


def write_evidence(rows: list[ExperimentRow], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "description": (
            "Synthetic local comparison. Bias is estimate minus the known pre-masking "
            "complete-data mean. Imputed cells are not treated as observations."
        ),
        "assumptions": [
            "MCAR: missingness probability does not depend on observed or missing values.",
            "MAR: conditional on modeled observed covariates, missingness does not depend on the unseen rating.",
            "MNAR: missingness depends on the latent rating; MAR-compatible estimators cannot guarantee unbiasedness.",
            "Rubin pooling is T = Ubar + (1 + 1/m) * B, with a Barnard-Rubin t reference distribution.",
        ],
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="ascii")
    return path
