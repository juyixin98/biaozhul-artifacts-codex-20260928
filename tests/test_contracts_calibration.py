"""Contract failure classes and fit/eval calibration behavior."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from app.calibration import fit_isotonic
from app.contracts import RunRequest, Sample
from app.errors import ContractViolation
from tests.reference_stats import weighted_pava_transform


def sample_dict(sample_id, y, p, w=1.0):
    return {"id": sample_id, "y": y, "p": p, "w": w}


@pytest.mark.parametrize(
    ("payload", "expected_fragment"),
    [
        (sample_dict("bad", 1, 1.01), "p must be finite"),
        (sample_dict("bad", 1, -0.01), "p must be finite"),
        (sample_dict("bad", 1, float("nan")), "p must be finite"),
        (sample_dict("bad", 1, 0.5, 0.0), "w must be finite"),
        (sample_dict("bad", 2, 0.5), "Input should be less than or equal to 1"),
    ],
)
def test_sample_contract_rejects_invalid_values(payload, expected_fragment, run_identity):
    run_identity.emit("contract_invalid_sample", payload=payload, failure_category="VALIDATION_ERROR")
    with pytest.raises(ValidationError) as exc_info:
        Sample.model_validate(payload)
    assert expected_fragment in str(exc_info.value)


def test_probability_out_of_range_is_rejected_not_clipped(run_identity):
    run_identity.emit("kernel_probability_boundary", value=1.2, failure_category="ValueError")
    from app.kernel import fixed_bins

    with pytest.raises(ValueError, match="p must be finite"):
        fixed_bins([1.2], [1], [1.0])


def test_fit_and_eval_overlap_is_contract_violation(run_identity):
    run_identity.emit("fit_eval_overlap", overlap=["shared"])
    payload = {
        "fit": [sample_dict("shared", 1, 0.7), sample_dict("fit-only", 0, 0.2)],
        "eval": [sample_dict("shared", 1, 0.7), sample_dict("eval-only", 0, 0.3)],
    }
    with pytest.raises(ValidationError, match="fit and eval sample ids must be disjoint"):
        RunRequest.model_validate(payload)


def test_fixed_edges_must_start_and_end_on_unit_interval(run_identity):
    run_identity.emit("invalid_edges", edges=[0.1, 0.9])
    payload = {
        "fit": [sample_dict("f1", 0, 0.2)],
        "eval": [sample_dict("e1", 1, 0.8)],
        "fixed_bins": {"edges": [0.1, 0.9]},
    }
    with pytest.raises(ValidationError, match="must start at 0 and end at 1"):
        RunRequest.model_validate(payload)


def test_weighted_pava_matches_independent_reference_and_uses_fit_only(run_identity):
    fit_p = np.array([0.1, 0.2, 0.8, 0.9])
    fit_y = np.array([1, 0, 1, 0])
    fit_w = np.array([1.0, 1.0, 1.0, 1.0])
    eval_p = np.array([0.0, 0.15, 0.5, 0.85, 1.0])
    run_identity.emit(
        "calibrator_reference_compare",
        fit_n=len(fit_p),
        eval_n=len(eval_p),
        decision="eval probabilities never enter fit_p",
    )

    model = fit_isotonic(fit_p, fit_y, fit_w)
    actual = model.transform(eval_p)
    expected = weighted_pava_transform(
        fit_p.tolist(), fit_y.tolist(), fit_w.tolist(), eval_p.tolist()
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert np.all(actual >= 0.0)
    assert np.all(actual <= 1.0)


def test_calibrator_refuses_out_of_bounds_evaluation_probability(run_identity):
    run_identity.emit("calibrator_boundary", failure_category="VALIDATION_ERROR")
    model = fit_isotonic(np.array([0.1, 0.9]), np.array([0, 1]), np.array([1.0, 1.0]))
    with pytest.raises(ContractViolation, match="must be finite and within"):
        model.transform(np.array([-0.1]))
