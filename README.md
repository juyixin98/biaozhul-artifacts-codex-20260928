# Bounded Nonlinear Least Squares Trust-Region Service

A local FastAPI service for nonlinear least-squares fitting with finite or
one-sided parameter bounds. The implementation is split into expression
parsing, a normalized trust-region solver, numerical diagnostics, API
schemas, configuration, and independent tests.

## What It Implements

- Restricted mathematical models: `f(x, parameters) -> y`.
- Forward-mode automatic differentiation through dual numbers.
- Central and one-sided finite-difference derivative checks at the initial and
  final points.
- Explicit parameter scaling. Every trust-region step is taken in normalized
  coordinates `z`; physical parameters `theta` are recovered by the mapping
  layer.
- Bound-aware Steihaug/truncated-CG trust-region subproblems.
- Actual and predicted reduction, reduction ratio, CG termination reason,
  step norm, trust radius, and accept/reject decision for every iteration.
- SVD rank, singular values, condition number, and parameter correlation from
  the normalized Jacobian.
- Interior and boundary KKT stationary points are reported separately.

## Allowed Expressions

The expression parser uses an AST whitelist. It accepts:

- Arithmetic: `+`, `-`, `*`, `/`, unary signs, and power.
- The independent variable `x`.
- Declared parameter identifiers.
- Constants `pi` and `e` and finite numeric literals.
- Functions: `exp`, `expm1`, `log`, `log1p`, `sqrt`, `sin`, `cos`, `tan`,
  `atan`, and `arctan`.

Attribute access, imports, assignment, control flow, and arbitrary function
calls are rejected before evaluation.

## Parameter Scaling

Each parameter has a positive `scale`. It represents a meaningful physical
step, not merely display formatting.

- Both bounds finite: `theta = lower + scale*sinh(q*z)`, `0 <= z <= 1`, where
  `q = asinh((upper-lower)/scale)`.
- Lower bound only: `theta = lower + scale*sinh(z)`.
- Upper bound only: `theta = upper - scale*sinh(z)`.
- Unbounded: `theta = scale*z`.

The physical Jacobian is transformed with `dtheta/dz`, so Gauss-Newton
matrices and projected gradients refer to normalized parameter steps. The
chain-rule factors are independently tested against numerical differences.

## Run

No virtual environment is required in the supplied machine:

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Example solve:

```bash
curl -s -X POST http://127.0.0.1:8000/solve \
  -H 'Content-Type: application/json' \
  -H 'X-Run-ID: manual-exp-001' \
  -d '{
    "expression": "a*exp(-b*x)+c",
    "x": [0, 1, 2, 3, 4],
    "y": [3.5, 1.98, 1.24, 0.87, 0.68],
    "parameters": [
      {"name": "a", "initial": 0.1, "lower": 0, "upper": 10, "scale": 1},
      {"name": "b", "initial": 2.0, "lower": 0, "upper": 5, "scale": 0.2},
      {"name": "c", "initial": 1.0, "lower": -2, "upper": 5, "scale": 1}
    ]
  }'
```

Run the local synthetic demonstration without starting HTTP:

```bash
python3 scripts/demo.py
```

## Error Semantics

Failures are never encoded as success. The response status field is one of:

| Status | HTTP | Meaning |
| --- | --- | --- |
| `converged` | 200 | Projected gradient satisfied tolerance and the final Jacobian is full rank. |
| `nonconvergent` | 422 | Iteration, step, trust-radius, or subproblem termination occurred before the KKT tolerance. |
| `unidentifiable` | 422 | The final normalized Jacobian has numerical rank below the parameter count. |
| `illegal_function_value` | 422 | A prediction, residual, or Jacobian contains `NaN` or infinity, for example `log(non-positive)`. |
| `derivative_check_failed` | 422 | AD exceeds the configured finite-difference tolerance at the initial or final point. |
| `invalid_expression` | 400 | The expression is syntactically invalid or contains an AST node/identifier outside the whitelist. |
| `invalid_request` | 400/422 | Arrays, bounds, initial values, parameter names, or solver configuration are invalid. |
| `internal_error` | 500 | Unexpected bug; still returned with an explicit run ID and never as success. |

`boundary_stationary` means the projected gradient is small and at least one
parameter satisfies an active-bound KKT condition. `interior_stationary`
means the projected gradient is small without an active bound.

## Numerical Evidence

Each iteration records:

- `actual_reduction = old_objective - new_objective`
- `predicted_reduction` from the local quadratic
- `reduction_ratio = actual / predicted`
- normalized step norm
- trust radius before and after
- projected gradient norm
- CG iterations and terminal reason (`cg_converged`, `trust_boundary`,
  `parameter_boundary`, negative curvature, or Cauchy fallback)
- accept/reject decision

The service logs structured JSON events containing `run_id`, service,
Python, NumPy, and SciPy versions, derivative checks, iteration progress,
and the final decision basis. A client may provide `X-Run-ID`; if absent,
the service generates one.

## Tests And Reproduction

Run all tests:

```bash
python3 -m pytest
```

The tests use only local synthetic data. They include:

- Exponential decay from a bad initial point. The result is compared with
  SciPy `optimize.least_squares`, which is an independent reference solver and
  is not used by the production solver.
- Michaelis-Menten/saturation fitting in a strongly correlated low-x regime.
  Parameters and residuals are compared with the independent SciPy solution;
  correlation and condition number are asserted.
- Interior versus boundary stationary points.
- Explicit `nonconvergent`, `unidentifiable`, and `illegal_function_value`
  categories.
- Restricted parser rejection.
- AD versus numerical differences through the scaling map.
- HTTP status, run identity, and log content.

Dependencies are listed in `requirements.txt`. The project does not require a
database or external service.
