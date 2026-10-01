from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.fixtures import build_fixtures
from app.models import CovarianceRequest
from app.sample_contract import prepare_sample
from app.service import estimate
from app.statistical_kernel import ledoit_wolf_shrinkage


REPORT = ROOT / "reports" / "experiment_report.json"


def _payload(values: np.ndarray, method: str, **kwargs) -> CovarianceRequest:
    return CovarianceRequest(
        data=np.where(np.isnan(values), None, values).tolist(),
        method=method,
        **kwargs,
    )


def independent_ledoit_wolf(z: np.ndarray) -> dict[str, object]:
    """Reference formula written independently from the production estimator."""
    n, p = z.shape
    s = z.T @ z / n
    mu = float(np.trace(s) / p)
    target = mu * np.eye(p)
    x2 = z * z
    beta_bar = float(np.sum((x2.T @ x2) / n - s * s) / n)
    delta = float(np.sum((s - target) ** 2) / p)
    shrinkage = min(1.0, max(0.0, beta_bar / delta)) if delta else 0.0
    covariance = (1.0 - shrinkage) * s + shrinkage * target
    eig = np.linalg.eigvalsh(covariance)
    return {
        "shrinkage": shrinkage,
        "min_eigenvalue": float(eig[0]),
        "max_eigenvalue": float(eig[-1]),
        "condition_number": float(eig[-1] / eig[0]),
        "trace": float(np.trace(covariance)),
        "frobenius_norm": float(np.linalg.norm(covariance)),
    }


def statsmodels_tyler_reference(z: np.ndarray) -> dict[str, object]:
    from statsmodels.robust.covariance import cov_tyler_regularized

    ref = cov_tyler_regularized(
        z, start_cov=np.eye(z.shape[1]), shrinkage_factor=0.25, maxiter=500
    )
    eig = np.linalg.eigvalsh(ref.cov)
    p = z.shape[1]
    inv = np.linalg.pinv(ref.cov)
    distances = np.einsum("ij,jk,ik->i", z, inv, z, optimize=True)
    weights = np.where(distances > np.finfo(float).eps, p / distances, 0.0)
    return {
        "shrinkage": float(ref.shrinkage_factor),
        "iterations": int(ref.n_iter),
        "min_eigenvalue": float(eig[0]),
        "max_eigenvalue": float(eig[-1]),
        "condition_number": float(eig[-1] / eig[0]),
        "first_weight": float(weights[0]),
        "trace": float(np.trace(ref.cov)),
        "frobenius_norm": float(np.linalg.norm(ref.cov)),
    }


def run() -> dict[str, object]:
    fixtures = build_fixtures()
    report: dict[str, object] = {
        "numpy_version": np.__version__,
        "fixtures": {},
    }

    for key in ("high_dim_small_sample", "collinear_columns", "strong_outliers", "rescaled_outliers"):
        values = fixtures[key].data
        robust = estimate(
            _payload(
                values,
                "robust_tyler_shrinkage",
                shrinkage=0.25,
                maxIterations=1000,
                tolerance=1e-12,
            ),
            request_id=f"experiment-{key}",
        )[0]
        prepared = prepare_sample(
            values.tolist(), assume_centered=False, center_method="median"
        )
        ref = statsmodels_tyler_reference(prepared.standardized)
        robust_standardized = np.array(robust.covariance) / np.multiply.outer(
            prepared.scales, prepared.scales
        )
        robust_eig = np.linalg.eigvalsh(robust_standardized)
        from statsmodels.robust.covariance import cov_tyler_regularized

        ref_cov = cov_tyler_regularized(
            prepared.standardized,
            start_cov=np.eye(prepared.standardized.shape[1]),
            shrinkage_factor=0.25,
            maxiter=1000,
            eps=1e-13,
        ).cov
        reference_original = (
            prepared.scales[:, None] * ref_cov * prepared.scales[None, :]
        )
        reference_original_eig = np.linalg.eigvalsh(reference_original)
        report["fixtures"].__setitem__(
            key,
            {
                "description": fixtures[key].description,
                "shape": list(values.shape),
                "robust_tyler": {
                    "status": robust.status,
                    "rank": robust.rank,
                    "invertible": robust.invertible,
                    "iterations_from_steps": next(
                        step["iterations"]
                        for step in robust.steps
                        if step["step"] == "tyler.fixed_point"
                    ),
                    "shrinkage": robust.shrinkage,
                    "standardized_min_eigenvalue": float(robust_eig[0]),
                    "standardized_max_eigenvalue": float(robust_eig[-1]),
                    "standardized_condition_number": float(robust_eig[-1] / robust_eig[0]),
                    "original_min_eigenvalue": robust.eigenvalues[0],
                    "original_max_eigenvalue": robust.eigenvalues[-1],
                    "original_condition_number": robust.condition_number,
                    "first_weight": robust.outlier_weights[0],
                    "standardized_trace": float(np.trace(robust_standardized)),
                },
                "statsmodels_standardized_reference": ref,
                "statsmodels_original_reference": None
                if reference_original_eig is None
                else {
                    "min_eigenvalue": float(reference_original_eig[0]),
                    "max_eigenvalue": float(reference_original_eig[-1]),
                    "condition_number": float(
                        reference_original_eig[-1] / reference_original_eig[0]
                    ),
                },
            },
        )

    lw_values = fixtures["collinear_columns"].data
    lw_response = estimate(
        _payload(lw_values, "ledoit_wolf"), request_id="experiment-ledoit-wolf"
    )[0]
    lw_prepared = prepare_sample(
        lw_values.tolist(), assume_centered=False, center_method="mean"
    )
    lw_reference = independent_ledoit_wolf(lw_prepared.standardized)
    lw_standardized = np.array(lw_response.covariance) / np.multiply.outer(
        lw_prepared.scales, lw_prepared.scales
    )
    lw_standardized_eig = np.linalg.eigvalsh(lw_standardized)
    report["fixtures"]["ledoit_wolf_collinear"] = {
        "description": "Non-robust shrinkage oracle on the collinear fixture.",
        "shape": list(lw_values.shape),
        "production": {
            "rank": lw_response.rank,
            "invertible": lw_response.invertible,
            "shrinkage": lw_response.shrinkage,
            "standardized_min_eigenvalue": float(lw_standardized_eig[0]),
            "standardized_max_eigenvalue": float(lw_standardized_eig[-1]),
            "standardized_condition_number": float(
                lw_standardized_eig[-1] / lw_standardized_eig[0]
            ),
            "original_min_eigenvalue": lw_response.eigenvalues[0],
            "original_max_eigenvalue": lw_response.eigenvalues[-1],
            "original_condition_number": lw_response.condition_number,
        },
        "independent_formula_reference": lw_reference,
    }

    missing = fixtures["missing_values"].data
    missing_response = estimate(
        _payload(missing, "ledoit_wolf"), request_id="experiment-missing"
    )[0]
    report["fixtures"]["missing_values"] = {
        "description": fixtures["missing_values"].description,
        "status": missing_response.status,
        "missing_mask_count": sum(sum(row) for row in missing_response.missing_mask),
        "imputed_values": [
            row for row in missing_response.imputed_values if any(v is not None for v in row)
        ],
        "uncertainties": [item.model_dump() for item in missing_response.uncertainties],
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
