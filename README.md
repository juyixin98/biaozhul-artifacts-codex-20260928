# Model-Driven Backend Tester (MDBT)

A configurable state-machine, model-driven testing backend that generates
both legal and deliberately illegal operation sequences, drives a system
under test (SUT), classifies failures, minimises failing sequences, and
stores replayable evidence in SQLite behind a FastAPI reporting API.

Stack: Python 3.12, FastAPI, Pydantic, httpx, SQLite (stdlib), pytest.

## Layout

```
fixtures/order_machine.json      minimal input model (order lifecycle)
mdbt/
  errors.py                      four-category error contract + HTTP mapping
  spec.py                        model parser + strict validation
  contracts.py                   Command/Observation/Verdict/Step/Finding
  oracle.py                      pure per-resource model oracle
  scheduler.py                   seeded legal/illegal command generator + log
  executor.py                    async transports + predict/call/observe kernel
  replay.py                      deterministic replay from a fresh model/SUT
  reducer.py                     precondition-preserving greedy minimisation
  scriptgen.py                   stdlib-only minimal replay script generator
  storage.py                     SQLite evidence store
  campaign.py                    orchestration: guided scenarios + fuzz
  api.py                         FastAPI control/report interface
  services/common.py             SUT response envelope (no state logic)
  services/reference.py          CORRECT, independently specified reference SUT
  services/buggy.py              SUT with 3 planted stateful defects
tests/                           independent pytest suite (29 tests)
scripts/                         demo runner + curl examples + result generator
docs/REPRODUCE.md                step-by-step reproduction
docs/RESULTS.md                  real, generated run results
artifacts/                       SQLite DB + minimal replay scripts
requirements.lock                pinned versions
```

## The SUT contract

`POST /orders` with `{"resource_id": "...", "action": "..."}` returns an
envelope `{"ok": bool, "state": ..., "exists": bool, ...}` with HTTP status:
201 create, 200 accepted, 404 unknown resource, 400 unknown action,
409 state conflict, 422 malformed body, 5xx internal.

Order lifecycle:

```text
create -> draft --submit--> submitted --approve--> paid
              draft --submit--> submitted --cancel--> cancelled
                                  paid --cancel--> cancelled
                                  paid --deliver--> delivered --close--> closed
```

The reference service owns a hand-written transition table independent of
the test core and the JSON fixture; the fixture and the table are two
separate descriptions of the same intended behaviour.

The buggy service adds exactly three illegal edges:

1. `closed --close--> closed` (repeat close accepted),
2. `paid --submit--> submitted` (out-of-order submit, rollback),
3. `delivered --cancel--> cancelled` (cancel after delivery).

## Quick start

```bash
python3 -m pytest tests -q
python3 scripts/generate_results.py
bash scripts/run_demo.sh          # in another shell
bash scripts/call_api_examples.sh
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for full details and
[docs/RESULTS.md](docs/RESULTS.md) for a saved real run (0 false positives
on the reference; exactly 3 `state_conflict` findings on the buggy SUT).

## Guarantees against "looks fine, computes wrong"

* Prediction never touches the SUT and never mutates the model; state moves
  only after the observation (`mdbt/executor.py::Executor.step`).
* Per-resource model state means interleaved resources cannot contaminate
  each other (covered by `test_interleaved_resources_have_independent_states`).
* Every random choice is recorded with RNG state, candidate set and reason;
  same seed reproduces identical summaries.
* Minimisation revalidates every deletion on a fresh SUT against a signature
  containing the model prediction, observation, category and states.
* Illegal inputs are intentionally generated in three flavours
  (illegal edge, unknown action, missing resource) and the four error
  categories never collapse into one another.
