# MCMC Autocorrelation, ESS, and Convergence Diagnostics

Local FastAPI service for one-chain and multi-chain numerical MCMC samples.
The implementation uses only NumPy/SciPy in its statistical kernel; SQLite is
the local result store. Statsmodels and ArviZ are used solely by independent
tests and the reproduction experiment as mature external references.

## Module Layout

| Path | Responsibility |
| --- | --- |
| `app/mcmc_diagnostics/models.py` | Pydantic sample and result contract. |
| `app/mcmc_diagnostics/kernel.py` | ACF, variance components, ESS, and R-hat calculations. |
| `app/mcmc_diagnostics/versioning.py` | Stable schema/kernel/algorithm metadata and request IDs. |
| `app/mcmc_diagnostics/repository.py` | SQLite persistence for successful and failed runs. |
| `app/mcmc_diagnostics/logging_config.py` | Request-linked local structured logs. |
| `app/main.py` | FastAPI routes, structured errors, headers, and persistence wiring. |
| `experiments/generate_fixtures.py` | Fixed-seed synthetic independent, AR(1), constant, shifted, and short fixtures. |
| `experiments/reproduce.py` | Independent theory/statsmodels/ArviZ comparison and machine-readable report. |
| `tests/` | Numerical, edge-category, contract, HTTP, and SQLite tests. |

## Statistical Rules

For each chain, the displayed ACF uses the centered, biased autocovariance

`C[t] = sum_i (x_i - mean(x))(x_i+t - mean(x)) / n`

and then divides all displayed lags by that same chain's `C[0]`. A constant
chain has zero variance, so post-lag-zero ACF is returned as `null`, not `1`.

The displayed lag limit is only a reporting truncation. The default is
`min(n-1, max(20, ceil(10*log10(n))))`; it is never used to assume later
correlation is zero. ESS separately scans autocorrelation pairs, stops at the
first non-positive Geyer pair, applies initial-monotone pair smoothing, and
computes `tau = -1 + 2 * sum(retained rho)` and `ESS = draws / tau`.

Three ESS views are returned:

- `mean`: unsplit-chain ESS for the mean parameter.
- `split_mean`: split-chain ESS for the mean, comparable to ArviZ's current
  `az.ess(..., method="mean")`.
- `bulk`: rank-normalized split-chain ESS, comparable to ArviZ `method="bulk"`.

Three R-hat values are returned:

- `classic`: Gelman-Rubin on whole balanced chains.
- `split`: whole-chain stationary check using two halves per chain.
- `rank_normalized`: maximum of bulk and folded-tail rank-normalized split
  R-hat, comparable to ArviZ `az.rhat(..., method="rank")`.

Within-chain variance `W` and between-chain variance `B` are reported
separately. Balanced ESS and R-hat are intentionally not estimated for unequal
chain lengths; weighted ANOVA components are still returned for inspection.

## Boundary Semantics

- Empty samples, missing values, infinities, and duplicate chain IDs fail the
  sample contract with a structured error.
- Fewer than 10 draws per analyzed series omits ESS with `SHORT_CHAIN_FOR_ESS`.
- Constant or partially constant aggregate input returns an explicit
  inconclusive diagnostic; it is never labeled as `n` independent draws.
- Zero within-chain variance with nonzero between-chain variance leaves R-hat
  undefined; differing constant chains are treated as a divergence signal.
- A single chain cannot provide genuine between-chain R-hat. Split-chain
  diagnostics help detect within-run trends but do not solve missing modes.

## Run The API

From the repository root:

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health and versions:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/version
```

Single-chain example:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/diagnostics/single \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-single","values":[1.2,1.5,1.3,1.8,1.4,2.0,1.9,1.6,2.2,2.1,1.8,2.4]}'
```

Multi-chain example:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/diagnostics/multi \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-multi","chains":[[1,1.2,0.8,1.1],[2.1,2.0,2.3,2.2],[1.2,1.1,1.4,1.3],[2.2,2.5,2.1,2.4]}'
```

Responses include `X-Request-Id`, `X-Kernel-Version`, and on success
`X-SQLite-Row-Id` / `X-SQLite-Path` headers. Retrieve a stored run with:

```bash
curl -s http://127.0.0.1:8000/api/v1/runs/demo-single
```

Logs are written to `logs/service.log`. Each line carries the request id,
event, endpoint, status, kernel version, and SQLite row id.

## Reproduction And Tests

Regenerate fixtures:

```bash
python3 experiments/generate_fixtures.py
```

Run the independent comparison:

```bash
python3 experiments/reproduce.py
```

It checks independent noise against statsmodels ACF and ArviZ ESS/R-hat, AR(1)
against both ArviZ and the AR(1) asymptotic integrated autocorrelation time,
constant chains for undefined categories, shifted chains for batch variance
separation, and short chains for explicit non-estimation. The report is written
to `experiments/results/report.json`.

Run the test suite:

```bash
pytest -q
```

## Configuration

Defaults live in `app/mcmc_diagnostics/config.py`. Environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MCMC_DIAG_DB` | `data/diagnostics.db` | SQLite database path. |
| `MCMC_DIAG_LOG` | `logs/service.log` | Log file path. |
| `MCMC_MIN_ESS_DRAWS` | `10` | Minimum series length before ESS is estimated. |
| `MCMC_RHAT_THRESHOLD` | `1.01` | R-hat warning threshold. |
| `MCMC_LOW_REL_ESS` | `0.10` | Low relative ESS threshold. |
| `MCMC_VERY_LOW_REL_ESS` | `0.01` | Severe ESS degradation threshold. |
| `MCMC_PERSIST` | `true` | Persist successful and failed runs. |

Exact dependency versions are pinned in `requirements.txt`.

## Interpretation Limits

R-hat near one and high ESS are not proof of convergence. They only fail to
detect selected finite-sample mixing problems in the supplied draws. They do
not validate the target distribution, initialization spread, mode coverage,
burn-in, gradient/divergence behavior, or stationarity. Long-memory processes,
short chains, nonstationary trends, and multimodal targets require longer runs
and additional diagnostics.
