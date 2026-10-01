# Synthetic Survey Imputation Comparison

This is a local FastAPI case study for comparing:

- the known complete-data baseline,
- complete-case analysis,
- deterministic mean imputation,
- Bayesian normal-regression multiple imputation.

Every participant and dataset is synthetic. The project does not require a
production account, database server, or real survey data.

## Project Layout

- `src/survey_imputation/contracts.py`: request/response contracts and validation.
- `src/survey_imputation/masking.py`: missing-mask invariants.
- `src/survey_imputation/kernel.py`: statistical estimators and Rubin pooling.
- `src/survey_imputation/synthetic.py`: known-parameter synthetic survey generator.
- `src/survey_imputation/experiment.py`: reproducible MCAR/MAR/MNAR comparison.
- `src/survey_imputation/repository.py`: SQLite versioned result storage.
- `src/survey_imputation/api.py`: FastAPI service.
- `tests/fixtures/minimal_survey.json`: hand-authored numeric fixture.
- `scripts/run_experiment.py`: command-line experiment entry point.
- `results/`: generated, inspectable evidence file.

## Missing Data Rules

The outcome is `rating`. `None` or `NaN` means missing. A numeric `0` is a
real structural zero: the synthetic questionnaire assigns zero to participants
who did not use support services. Missingness is only applied to support users.
The mask is cross-validated so a zero can never be silently treated as missing.

Training parameters use allowed observed rows only. Complete rows needed by a
method are used as observations; missing rows are not used to fit parameters.
Imputed values remain marked as imputed and are never appended as if they were
new true observations.

## Statistical Method

Mean imputation fills each missing rating with the arithmetic mean of observed
ratings. Its point estimate equals the complete-case mean in this case, while
the completed-data variance is typically too small.

Multiple imputation uses one normal linear regression model per completed
dataset:

1. Fit on observed ratings only.
2. Draw residual variance from a posterior chi-square distribution.
3. Draw regression coefficients conditional on that variance.
4. Draw missing outcomes independently, including an outcome-level residual.
5. Clip draws to the valid rating range `[0, 10]`.

Each of the `m` datasets gets an independent `numpy` random stream spawned from
the request seed. Rubin's rules pool the estimates:

```text
Qbar = mean(Q_l)
Ubar = mean(U_l)
B    = sample variance of Q_l with denominator m - 1
T    = Ubar + (1 + 1/m) * B
```

The service uses a t interval with Barnard-Rubin adjusted degrees of freedom.

## Assumptions And Limits

MCAR assumes missingness does not depend on observed or unobserved values.
MAR assumes that, conditional on modeled observed covariates, missingness does
not depend on the unseen rating. MAR cannot be proved from an observed dataset
alone.

Under MNAR, missingness may depend on the hidden outcome itself. The included
MNAR experiment demonstrates that MAR-compatible mean or regression imputation
can remain biased. No unbiasedness guarantee is made there; sensitivity
analysis or an explicit MNAR model would be required.

## Run The Tests

Use the provided system Python environment:

```bash
pytest -q
```

## Run The Reproducible Experiment

```bash
PYTHONPATH=src python3 scripts/run_experiment.py --n 1000 --m 30 --seed 20260930
```

This writes:

- SQLite results under `data/survey.db`;
- a JSON evidence report under `results/comparison_seed_20260930_n_1000.json`.

The current checked-in evidence was generated with that command. Re-running with
the same arguments reproduces the estimates exactly.

## Start The API

```bash
PYTHONPATH=src python3 -m uvicorn survey_imputation.api:app --reload
```

The default base URL is `http://127.0.0.1:8000`. Open `/docs` for the
interactive schema, or inspect `examples/api_examples.sh`.

Example endpoints:

- `GET /health`
- `GET /versions`
- `POST /datasets/generate`
- `POST /analyses`
- `GET /analyses/{request_id}`

## Diagnostics And Sensitive Values

Every response carries a `request_id`. Diagnostics use codes, accept/reject/
undetermined statuses, and state fields such as record counts, missing counts,
structural-zero counts, method names, and whether imputed values are treated as
observations. Logs include the request id and state, but participant ids in
multiple-imputation previews are rendered in redacted form such as `me***04`.
The API does not log free-text participant data because the contract contains
none.

## Dependency Locking

`requirements.lock` records the versions used with the supplied runtime. The
project also declares runtime dependencies in `pyproject.toml`.
