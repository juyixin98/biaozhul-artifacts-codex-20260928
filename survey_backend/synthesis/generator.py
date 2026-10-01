from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from survey_backend.models import SurveyRecord


@dataclass(frozen=True)
class TrueParameters:
    intercept: float = 2.8
    beta_q1: float = -0.35
    beta_q2: float = 0.55
    sigma: float = 0.75


@dataclass(frozen=True)
class SyntheticSurvey:
    record_ids: tuple[str, ...]
    q1: np.ndarray
    q2: np.ndarray
    y: np.ndarray
    observed_q1: np.ndarray
    observed_q2: np.ndarray
    eligible: np.ndarray
    missing_mask: np.ndarray
    mechanism: str
    true_parameters: TrueParameters
    seed: int

    def to_records(self) -> list[SurveyRecord]:
        records: list[SurveyRecord] = []
        for i, record_id in enumerate(self.record_ids):
            q1 = None if not self.observed_q1[i] else float(self.q1[i])
            q2 = None if not self.observed_q2[i] else float(self.q2[i])
            records.append(
                SurveyRecord(
                    record_id=record_id,
                    q1_screen_hours=q1,
                    q2_sleep_hours=q2,
                    satisfaction=float(self.y[i]),
                    eligible=bool(self.eligible[i]),
                )
            )
        return records


def _missing_indicators(
    q1: np.ndarray,
    q2: np.ndarray,
    y: np.ndarray,
    mechanism: str,
    probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = q1.size
    missing_q1 = np.zeros(n, dtype=bool)
    missing_q2 = np.zeros(n, dtype=bool)
    if mechanism == "complete":
        return missing_q1, missing_q2
    if mechanism == "mcar":
        score = np.full(n, probability)
    elif mechanism == "mar":
        # Probability depends on observed variables, including the other
        # feature and the satisfaction value, not on the missing value itself.
        score = expit(
            np.log(probability / max(1.0 - probability, 1e-6))
            - 0.45 * (q2 - np.mean(q2))
            + 0.30 * (y - np.mean(y))
        )
        missing_q2 = rng.random(n) < expit(
            -2.5 + 0.55 * (q1 - np.mean(q1)) + 0.30 * (y - np.mean(y))
        )
    elif mechanism == "mnar":
        # Probability directly depends on the potentially missing q1 itself.
        score = expit(
            np.log(probability / max(1.0 - probability, 1e-6))
            + 0.95 * (q1 - np.mean(q1))
        )
        missing_q2 = rng.random(n) < expit(
            -2.6 - 0.55 * (q2 - np.mean(q2)) + 0.25 * (y - np.mean(y))
        )
    else:
        raise ValueError(f"unsupported missingness mechanism: {mechanism}")
    missing_q1 |= rng.random(n) < score
    return missing_q1, missing_q2


def generate_survey(
    n: int = 320,
    *,
    seed: int = 20260930,
    mechanism: str = "mar",
    missing_probability: float = 0.20,
    ineligible_probability: float = 0.10,
    parameters: TrueParameters | None = None,
    make_q1_all_missing: bool = False,
) -> SyntheticSurvey:
    if n < 20:
        raise ValueError("synthetic survey requires at least 20 rows")
    if not 0.0 <= missing_probability < 0.8:
        raise ValueError("missing_probability must be in [0, 0.8)")
    if not 0.0 <= ineligible_probability < 0.8:
        raise ValueError("ineligible_probability must be in [0, 0.8)")
    params = parameters or TrueParameters()
    rng = np.random.default_rng(seed)

    q1_raw = rng.normal(3.2, 1.3, size=n)
    q1 = np.clip(q1_raw, 0.0, None)
    q1[rng.random(n) < 0.05] = 0.0
    q2 = rng.normal(7.0, 1.1, size=n)
    q2 = np.clip(q2, 0.0, 11.0)
    q2[rng.random(n) < 0.04] = 0.0
    noise = rng.normal(0.0, params.sigma, size=n)
    y = params.intercept + params.beta_q1 * q1 + params.beta_q2 * q2 + noise

    eligible = rng.random(n) >= ineligible_probability
    # Ineligible rows remain present for auditing but carry distinctive values.
    q1 = np.where(eligible, q1, 30.0)
    y = np.where(eligible, y, 100.0)

    missing_q1, missing_q2 = _missing_indicators(
        q1=q1,
        q2=q2,
        y=y,
        mechanism=mechanism,
        probability=missing_probability,
        rng=rng,
    )
    # Missingness applies only to observations allowed for estimation.
    missing_q1 &= eligible
    missing_q2 &= eligible
    if make_q1_all_missing:
        missing_q1[:] = eligible

    mask = np.column_stack([missing_q1, missing_q2, np.zeros(n, dtype=bool)])
    ids = tuple(f"P{i + 1:04d}" for i in range(n))
    return SyntheticSurvey(
        record_ids=ids,
        q1=q1,
        q2=q2,
        y=y,
        observed_q1=~missing_q1,
        observed_q2=~missing_q2,
        eligible=eligible,
        missing_mask=mask,
        mechanism=mechanism,
        true_parameters=params,
        seed=seed,
    )
