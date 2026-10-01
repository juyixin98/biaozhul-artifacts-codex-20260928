"""Synthetic survey data with known generating parameters.

The case is fictional and local. A record represents a participant who either
used support services or did not. Non-users have a structural support rating
of exactly zero; users may have a missing rating. Thus data cleaning cannot
confuse "true zero" with "did not answer".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import Diagnostic, DiagnosticStatus, MissingnessMechanism, SurveyRecord


TRUE_PARAMETERS = {
    "intercept": 7.25,
    "age_z": 0.20,
    "member": 0.45,
    "used_support": -0.65,
    "contact_frequency": 0.08,
    "noise_sd": 0.85,
    "p_used_support": 0.75,
    "p_member_when_used": 0.70,
    "p_member_when_not_used": 0.35,
}


@dataclass(frozen=True)
class GeneratedData:
    records: list[SurveyRecord]
    mechanism: MissingnessMechanism
    seed: int
    n: int
    n_missing: int
    true_mean_rating: float
    complete_baseline_mean: float
    parameters: dict[str, float]
    diagnostics: list[Diagnostic]


def _missing_probabilities(
    mechanism: MissingnessMechanism,
    eligible: np.ndarray,
    age_z: np.ndarray,
    member: np.ndarray,
    contact_frequency: np.ndarray,
    latent_rating: np.ndarray,
    target_probability: float,
) -> np.ndarray:
    p = np.zeros(eligible.shape, dtype=float)
    if mechanism == MissingnessMechanism.complete or not np.any(eligible):
        return p

    eligible_fraction = float(np.mean(eligible))
    eligible_target = min(0.95, target_probability / eligible_fraction)
    if mechanism == MissingnessMechanism.mcar:
        p[eligible] = eligible_target
    elif mechanism == MissingnessMechanism.mar:
        # Missingness depends on observed covariates, not the unseen rating.
        logits = -1.25 + 0.85 * age_z + 0.55 * member + 0.18 * contact_frequency
        raw = 1.0 / (1.0 + np.exp(-logits))[eligible]
        p[eligible] = eligible_target * raw / float(np.mean(raw))
    elif mechanism == MissingnessMechanism.mnar:
        # Lower latent support ratings are more likely to be missing.
        logits = 1.35 - 0.55 * latent_rating + 0.25 * age_z + 0.20 * member
        raw = 1.0 / (1.0 + np.exp(-logits))[eligible]
        p[eligible] = eligible_target * raw / float(np.mean(raw))
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ValueError(f"unknown missingness mechanism: {mechanism}")
    return np.clip(p, 0.0, 0.98)


def generate_survey(
    n: int = 1000,
    mechanism: MissingnessMechanism | str = MissingnessMechanism.complete,
    missing_probability: float = 0.20,
    seed: int = 20260930,
    request_id: str = "synthetic-case",
    missing_seed: int | None = None,
) -> GeneratedData:
    mechanism = MissingnessMechanism(mechanism)
    outcome_rng = np.random.default_rng(seed)
    missing_rng = np.random.default_rng(seed if missing_seed is None else missing_seed)

    age_z = np.clip(outcome_rng.normal(0.0, 1.0, n), -3.0, 3.0)
    used_support = (outcome_rng.random(n) < TRUE_PARAMETERS["p_used_support"]).astype(int)
    member_probability = np.where(
        used_support == 1,
        TRUE_PARAMETERS["p_member_when_used"],
        TRUE_PARAMETERS["p_member_when_not_used"],
    )
    member = (outcome_rng.random(n) < member_probability).astype(int)
    contact_frequency = np.clip(
        outcome_rng.poisson(np.where(used_support == 1, 3.0, 0.7)), 0, 10
    ).astype(float)

    latent = (
        TRUE_PARAMETERS["intercept"]
        + TRUE_PARAMETERS["age_z"] * age_z
        + TRUE_PARAMETERS["member"] * member
        + TRUE_PARAMETERS["used_support"] * used_support
        + TRUE_PARAMETERS["contact_frequency"] * contact_frequency
        + outcome_rng.normal(0.0, TRUE_PARAMETERS["noise_sd"], n)
    )
    # Non-users have a deliberate structural zero. Users' support ratings
    # remain positive on the continuous scale.
    latent = np.where(used_support == 1, np.maximum(latent, 0.05), 0.0)
    complete_ratings = np.round(np.clip(latent, 0.0, 10.0), 3)

    eligible = used_support == 1
    probabilities = _missing_probabilities(
        mechanism,
        eligible,
        age_z,
        member,
        contact_frequency,
        latent,
        missing_probability,
    )
    missing = eligible & (missing_rng.random(n) < probabilities)
    ratings = np.where(missing, np.nan, complete_ratings)

    records = [
        SurveyRecord(
            record_id=f"rec-{index + 1:05d}",
            age_z=float(age_z[index]),
            member=int(member[index]),
            used_support=int(used_support[index]),
            contact_frequency=float(contact_frequency[index]),
            rating=None if np.isnan(ratings[index]) else float(ratings[index]),
        )
        for index in range(n)
    ]
    n_true_zero = int(np.sum(complete_ratings == 0.0))
    diagnostics = [
        Diagnostic(
            code="synthetic_local_fixture",
            status=DiagnosticStatus.accepted,
            message="All rows are locally generated; no production account or participant data is used.",
            observed_state={"request_id": request_id, "n_records": n, "seed": seed},
        ),
        Diagnostic(
            code="zero_missing_separation",
            status=DiagnosticStatus.accepted,
            message="Non-user support ratings are exact zeros and missingness is applied only to users.",
            observed_state={
                "n_true_zero": n_true_zero,
                "n_missing": int(missing.sum()),
                "missing_rows_are_users": bool(np.all((used_support == 1) | ~missing)),
            },
        ),
        Diagnostic(
            code="mar_assumption_status",
            status=(
                DiagnosticStatus.accepted
                if mechanism == MissingnessMechanism.mar
                else DiagnosticStatus.undetermined
            ),
            message=(
                "This generator creates MAR missingness from observed covariates, but an observed "
                "dataset alone cannot prove MAR. MNAR dependence on the unseen rating can bias imputation."
                if mechanism == MissingnessMechanism.mar
                else "MAR is not empirically identifiable from observed data alone."
            ),
            observed_state={"mechanism": mechanism.value, "mnar_unbiased_guaranteed": False},
        ),
    ]
    return GeneratedData(
        records=records,
        mechanism=mechanism,
        seed=seed,
        n=n,
        n_missing=int(missing.sum()),
        true_mean_rating=float(np.mean(complete_ratings)),
        complete_baseline_mean=float(np.mean(complete_ratings)),
        parameters=dict(TRUE_PARAMETERS),
        diagnostics=diagnostics,
    )
