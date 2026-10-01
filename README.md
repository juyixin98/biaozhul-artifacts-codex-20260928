# Local Two-Process Consensus ADMM

This repository contains a local teaching backend for consensus ADMM over a
separable, convex, quadratic objective with exactly two compute processes:

```text
minimize  sum_i 0.5 * x_i.T @ P_i @ x_i + q_i.T @ x_i + r_i
subject to x_1 = z, x_2 = z
```

The two workers have the same consensus dimension but can use different
matrices and different `sample_count` metadata. All data is local synthetic
data; no external accounts or business datasets are required.

## Module Layout

- `consensus_admm/parser.py`: validates dimensions, finite values, symmetry
  and positive semidefinite quadratic matrices.
- `consensus_admm/kernel.py`: pure NumPy/SciPy ADMM operations. It enforces the
  fixed order `local x -> consensus z -> scaled dual u`.
- `consensus_admm/reference.py`: independent centralized reference solve used
  for diagnostics and cross-checking.
- `consensus_admm/service.py`: round state machine, submission contracts,
  timeout and resume handling, residual decisions.
- `consensus_admm/state.py` and `storage.py`: event reducer, JSONL evidence
  logs, atomic snapshots and local file locks.
- `consensus_admm/evidence.py`: canonical hashes and round tokens.
- `consensus_admm/api.py`: FastAPI service entry and response serialization.
- `consensus_admm/worker.py`: a local worker process that polls tasks, solves
  its quadratic update and submits evidence.
- `tests/`: independent assertions for the math, service behavior, recovery,
  rho adjustment, timeout, error categories and non-convergence.

## Dependencies

The environment is expected to provide Python 3 with these globally installed
packages. The exact versions used during delivery are locked in
`requirements.txt`:

```text
numpy, scipy, fastapi, pydantic, httpx, uvicorn, pandas, pytest
```

Do not create a virtual environment for the provided machine; run Python as
`python3`.

## Run The Service

```bash
ADMM_DATA_DIR=.local_data/admm \
python3 -m uvicorn consensus_admm.api:app --host 127.0.0.1 --port 8000
```

Optional server settings:

- `ADMM_DATA_DIR`: run storage directory, default `.local_data/admm`
- `ADMM_MAX_RUNS`: stored run count cap, default `256`
- `ADMM_MAX_PROBLEM_DIM`: maximum variable dimension, default `256`
- `ADMM_MAX_ITERATIONS`: maximum accepted iterations, default `1000`
- `ADMM_MAX_EVENT_LOG_BYTES`: per-run JSONL cap, default `67108864`

Create the bundled example:

```bash
curl -sS -X POST http://127.0.0.1:8000/runs \
  -H 'Content-Type: application/json' \
  --data @examples/create_run.json
```

Copy the returned `run_id`, then start one process for each worker:

```bash
python3 -m consensus_admm.worker \
  --base-url http://127.0.0.1:8000 \
  --run-id RUN_ID --worker-id worker-a

python3 -m consensus_admm.worker \
  --base-url http://127.0.0.1:8000 \
  --run-id RUN_ID --worker-id worker-b
```

Inspect state and evidence:

```bash
curl -sS http://127.0.0.1:8000/runs/RUN_ID/diagnostics
curl -sS http://127.0.0.1:8000/runs/RUN_ID/trajectory
curl -sS http://127.0.0.1:8000/runs/RUN_ID/events
```

Export a trajectory as CSV:

```bash
python3 scripts/export_trajectory.py http://127.0.0.1:8000 RUN_ID trajectory.csv
```

## ADMM Contract

The scaled dual is `u = y / rho`. Each round uses one immutable round token
covering `run_id`, iteration, `rho`, `z_pre`, `u_pre` and problem digest.

1. Open the round and choose `rho_k`. If rho changed, rescale every dual with
   `u <- u * rho_old / rho_new`, preserving `rho * u`.
2. Both workers compute local updates from the same prior round:
   `x_i = (P_i + rho I)^-1 (rho z - u_i - q_i)`.
3. After both submissions are present, update consensus:
   `z = mean(x_i)`.
4. Update dual variables:
   `u_i <- u_i + x_i - z`.
5. Evaluate primal and dual residual separately.

Primal residual:

```text
norm(concat(x_1-z, x_2-z))
```

Dual residual:

```text
sqrt(2) * rho * norm(z - z_previous)
```

The run converges only when both independent boolean checks pass:

```text
primal_residual <= eps_primal
dual_residual   <= eps_dual
```

If the iteration limit is reached, the API remains explicit about whether
`primal`, `dual`, or both conditions are still false.

## Timeout And Recovery

A round cannot advance until both worker submissions are present. If the
deadline passes, a late submission records no round update; the coordinator
marks the round as `timed_out`. Existing valid submissions are retained.

```bash
curl -sS -X POST http://127.0.0.1:8000/runs/RUN_ID/timeout \
  -H 'Content-Type: application/json' \
  -d '{"reason":"worker-a-unreachable"}'

curl -sS -X POST http://127.0.0.1:8000/runs/RUN_ID/resume \
  -H 'Content-Type: application/json' \
  -d '{"deadline_ms":5000,"reason":"retry-missing-worker"}'
```

The resumed round keeps the same iteration, `z_pre`, dual values and round
token. Submissions carrying a token from an older or newer round are rejected
as computation-contract failures.

## Error Categories

Responses use a stable nested shape:

```json
{"error":{"category":"...", "code":"...", "message":"...", "details":{}}}
```

- `input_error` (`400` or `404`): malformed model, wrong dimensions, unknown
  run or worker.
- `state_conflict` (`409`): duplicate submission, terminal run, deadline not
  elapsed, timed-out round without resume.
- `resource_exhausted` (`507`): dimension, iteration, stored-run or event-log
  limit.
- `computation_failure` (`422`): bad round token, mismatched rho/z/u,
  inconsistent objective or stationarity evidence.
- `computation_failure` (`500`): internal kernel numerical failure.

## Persistence And Replay

Each run is stored under `$ADMM_DATA_DIR/runs/RUN_ID/`:

- `events.jsonl`: append-only evidence stream with sequence numbers.
- `state.snapshot.json`: atomically replaced reducer snapshot.
- `.lock`: process-local coordination lock.

Important event types are `run_created`, `round_opened`,
`submission_received`, `submission_rejected`, `round_timed_out`,
`round_resumed` and `round_completed`. Restarting the service replays events
after the latest snapshot and resumes the exact same iteration and round
inputs.

## Verification

Run the independent test suite:

```bash
pytest -q
```

The tests include:

- asymmetric worker matrices and sample counts compared with an independently
  aggregated central linear solve;
- rho schedule changes with equality of pre/post unscaled duals;
- separate primal and dual termination booleans;
- a whole-round timeout with one worker missing and retained recovery state;
- stale round-token rejection;
- service restart from disk;
- max-iteration non-convergence classification;
- distinct input, state-conflict, resource-limit and computation-failure
  categories.

Delivery was verified on this machine with `13 passed`. A real two-process
run also converged in 53 rounds with run id
`d0f59801325340f7a838a716b6e8cc9a`; its full trajectory is in
`.demo-data2/clean-verification-report.json`.

Final values from that run:

```text
status                 converged
completed rounds       53
submissions            106
max point error        8.1178448963e-09
primal residual        3.3435971534e-08 <= 3.8153292782e-08
dual residual          4.6671473354e-09 <= 7.1412431298e-08
objective gap          1.1386146959e-07
```

## Remaining Limitations

- Scope is intentionally exactly two workers; general `N` consensus is not
  exposed through the HTTP contract.
- Local worker matrices must be PSD, and the aggregate problem must have a
  finite minimizer.
- Only explicit rho schedules are supported; adaptive rho policies are not
  automatically inferred.
- The included worker is a polling CLI, not a remote task queue or production
  process supervisor.
- Authentication, authorization and multi-tenant quotas are outside this
  local teaching backend.
