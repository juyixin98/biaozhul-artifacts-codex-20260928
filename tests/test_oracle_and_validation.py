"""Input model validation + pure oracle boundary behaviour."""
from __future__ import annotations

import pytest

from mdbt.errors import ErrorCategory, InputError
from mdbt.oracle import FSMModel
from mdbt.spec import load_spec, load_spec_file


def test_fixture_loads_with_expected_states_and_actions():
    spec = load_spec_file("fixtures/order_machine.json")
    assert spec.initial_state == "draft"
    assert spec.states == {
        "draft", "submitted", "paid", "delivered", "closed", "cancelled"
    }
    assert spec.terminal_states == {"closed", "cancelled"}
    # create is a first-class construction op, not an edge
    assert spec.legal_actions("draft") == ["submit"]


@pytest.mark.parametrize(
    "mutator, fragment",
    [
        (lambda d: d.update(initial_state="nope"), "initial_state"),
        (lambda d: (d.update(states=["draft"]), d.update(terminal_states=[])),
         "must be a non-empty array"),
        (lambda d: d["states"].append("draft"), "duplicates"),
        (lambda d: d["transitions"].append(d["transitions"][0]), "two transitions"),
        (lambda d: d["transitions"][0].update(action="explode"), "unknown action"),
        (lambda d: d["transitions"][0].update(to="nowhere"), "unknown state"),
    ],
)
def test_bad_models_are_rejected_as_input_error(mutator, fragment):
    import json

    raw = json.loads(open("fixtures/order_machine.json").read())
    mutator(raw)
    with pytest.raises(InputError) as exc:
        load_spec(raw)
    assert exc.value.category == ErrorCategory.INPUT_ERROR
    assert fragment in exc.value.message or fragment in str(exc.value.detail)


def test_unreachable_state_is_rejected():
    raw = {
        "name": "x",
        "initial_state": "a",
        "states": ["a", "b", "ghost"],
        "actions": ["go"],
        "transitions": [{"action": "go", "from": "a", "to": "b"}],
    }
    with pytest.raises(InputError) as exc:
        load_spec(raw)
    assert exc.value.category == ErrorCategory.INPUT_ERROR
    assert exc.value.detail["unreachable"] == ["ghost"]


def test_expected_does_not_mutate_model(spec):
    model = FSMModel.fresh(spec)
    before = dict(model.states)
    prediction = model.expected("r1", "close", resource_exists=True)
    assert prediction.accepted is False
    assert prediction.category == ErrorCategory.STATE_CONFLICT
    # repeated calls with different actions must observe the same state
    for action in ("close", "cancel", "deliver", "approve", "submit"):
        model.expected("r1", action, resource_exists=True)
    assert dict(model.states) == before


def test_oracle_distinguishes_missing_resource_unknown_action_illegal_edge(spec):
    model = FSMModel.fresh(spec)
    missing = model.expected("r1", "submit", resource_exists=False)
    assert (missing.accepted, missing.category) == (False, ErrorCategory.INPUT_ERROR)

    unknown = model.expected("r1", "teleport", resource_exists=True)
    assert (unknown.accepted, unknown.category) == (False, ErrorCategory.INPUT_ERROR)

    model.states["r1"] = "draft"
    illegal = model.expected("r1", "close", resource_exists=True)
    assert (illegal.accepted, illegal.category) == (
        False,
        ErrorCategory.STATE_CONFLICT,
    )

    legal = model.expected("r1", "submit", resource_exists=True)
    assert (legal.accepted, legal.to_state, legal.category) == (
        True, "submitted", None
    )


def test_interleaved_resources_have_independent_states(spec):
    model = FSMModel.fresh(spec)
    # r1 reaches paid while r2 stays draft; predictions must not bleed across.
    model.states["r1"] = "paid"
    model.states["r2"] = "draft"
    assert model.expected("r2", "submit", resource_exists=True).accepted is True
    assert (
        model.expected("r2", "deliver", resource_exists=True).category
        == ErrorCategory.STATE_CONFLICT
    )
    assert model.expected("r1", "deliver", resource_exists=True).accepted is True


def test_create_semantics(spec):
    model = FSMModel.fresh(spec)
    create_new = model.expected("r1", "create", resource_exists=False)
    assert create_new.accepted is True
    assert create_new.to_state == "draft"
    create_again = model.expected("r1", "create", resource_exists=True)
    assert create_again.accepted is False
    assert create_again.category == ErrorCategory.STATE_CONFLICT


def test_apply_observation_only_advances_on_observed_state(spec):
    model = FSMModel.fresh(spec)
    # rejected + same state => no movement, no resync
    new_state, resynced = model.apply_observation(
        "r1", "close",
        observed_accepted=False,
        observed_state="draft",
        resource_exists_after=True,
    )
    assert new_state == "draft" and resynced is False

    # accepted with different state => follow observation
    new_state, resynced = model.apply_observation(
        "r1", "submit",
        observed_accepted=True,
        observed_state="submitted",
        resource_exists_after=True,
    )
    assert new_state == "submitted" and resynced is False

    # rejected but SUT reveals drift => resync and count it
    new_state, resynced = model.apply_observation(
        "r1", "close",
        observed_accepted=False,
        observed_state="closed",
        resource_exists_after=True,
    )
    assert new_state == "closed" and resynced is True
    assert model.resync_count == 1
