from __future__ import annotations

import json
from pathlib import Path

from survey_backend.synthesis.generator import generate_survey


def main() -> None:
    survey = generate_survey(
        n=60,
        seed=4242,
        mechanism="mar",
        missing_probability=0.18,
        ineligible_probability=0.10,
    )
    payload = {
        "dataset_id": "fixture-mar-60",
        "request_id": "fixture-request-4242",
        "seed": 4242,
        "imputations": 4,
        "burn_in": 5,
        "confidence": 0.95,
        "true_parameters": {
            "intercept": survey.true_parameters.intercept,
            "beta_q1": survey.true_parameters.beta_q1,
            "beta_q2": survey.true_parameters.beta_q2,
            "sigma": survey.true_parameters.sigma,
        },
        "records": [record.model_dump() for record in survey.to_records()],
    }
    target = Path("tests/fixtures/minimal_survey.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
