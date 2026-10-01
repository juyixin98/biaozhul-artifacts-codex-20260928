# Entropic Optimal Transport

This project solves finite discrete entropically regularized optimal transport
with Python, FastAPI, NumPy, and SciPy.

The optimizer minimizes

```text
<P, C> + epsilon * sum_ij P_ij (log(P_ij) - 1)
```

subject to `P 1 = a`, `P^T 1 = b`, and `P >= 0`. The API returns both the
unregularized transport cost `<P, C>` and the regularized objective.

## Project Layout

```text
entropic_ot/
  validation.py    # input contract, finite/non-negative/mass-balance checks
  solver.py        # log-domain Sinkhorn kernel and objective diagnostics
  diagnostics.py   # run IDs, stop reasons, sampled iteration evidence
  errors.py        # stable error categories
  schemas.py       # FastAPI/Pydantic data contracts
  service.py       # HTTP entrypoint and replay logging
tests/             # hand-calculated, LP-reference, boundary, and API tests
examples/          # replayable JSON request bodies
```

## Configuration

The solver uses `float64`. Important request fields are:

- `epsilon`: positive finite entropy regularization.
- `max_iterations`: positive integer, capped by the API at `1,000,000`.
- `atol`: absolute marginal tolerance. The default is `1e-10`.
- `rtol`: relative marginal tolerance. The default is `1e-8`.

The effective line tolerance is

```text
atol * max(total_mass, 1) + rtol * target_marginal
```

for both source and target marginals. Total source and target masses are
validated with a default tolerance `1e-12 + 1e-9 * total_mass`; mismatched
mass is rejected rather than silently normalized.

Set `ENTROPIC_OT_LOG_PATH` to a JSONL path to persist replay evidence. Each
success log contains the run ID, complete request payload, key result values,
iterations, two-sided marginal errors, stop reason, and sampled events.

## Clean Setup

From the project root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository was developed with Python 3.12.3 and the pinned versions in
`requirements.txt`.

## Run Tests

```bash
pytest -q
```

The test suite checks:

- an independently derived 2x2 closed-form plan and objective;
- large cost with tiny epsilon and explicit zero underflow edges;
- zero-mass rows and columns and the all-zero-mass problem;
- a 3x3 result against an independent SciPy `linprog` LP solution;
- budget exhaustion marked as infeasible rather than returned as a plan;
- distinct input, resource, and computation error categories;
- HTTP parsing, semantic validation, success, replay logs, and failures.

## Start The API

```bash
export ENTROPIC_OT_LOG_PATH="$PWD/logs/entropic-ot.jsonl"
uvicorn entropic_ot.service:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## Request Example

```bash
curl -s -X POST http://127.0.0.1:8000/v1/optimal-transport \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request_two_by_two.json
```

A success response includes:

- `plan`: matrix `P`, including exact zero rows and columns;
- `transport_cost`: `<P, C>` using the original input costs;
- `regularized_objective`: `transport_cost + epsilon * entropy_term`;
- `entropy_term`: `sum P_ij (log P_ij - 1)`;
- `kl_to_uniform`: diagnostic KL divergence over the full matrix cells;
- `diagnostics`: run ID, active shape, iteration count, two-sided errors,
  feasibility, stop reason, and sampled intermediate states.

## Error Contract

HTTP errors use stable machine-readable codes:

- `400 input_error`: model parsing or semantic input validation failure.
- `409 state_conflict`: conflicting service state.
- `503 resource_exhausted`: iteration budget exhausted; diagnostics explicitly
  report `feasible=false` and no plan is returned.
- `500 computation_failed`: overflow or non-finite numeric state.

## Numerical Strategy

The kernel updates dual potentials in the log domain with SciPy
`logsumexp`, avoiding direct `exp(-C/epsilon)` kernel formation. Costs are
row/column gauge-shifted so the active rows and columns each contain a zero.
This leaves the optimal coupling unchanged while preventing finite-range
problems when costs are large compared with epsilon.

Zero-mass rows and columns are removed from the active problem and assigned
exact zero entries in the returned plan. Both marginal sides are checked
after every sweep and again after materializing `P=exp(log P)`.

