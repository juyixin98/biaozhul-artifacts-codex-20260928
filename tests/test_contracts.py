from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mbt_server.core.domain import RunRequest, StateMachineModel
from mbt_server.core.scheduler import create_plan


def test_rejected_transition_must_remain_in_same_state(order_model: dict) -> None:
    order_model["transitions"][5]["next_state"] = "draft"
    with pytest.raises(ValidationError) as exc:
        StateMachineModel.model_validate(order_model)
    assert "rejected operation must leave model state unchanged" in str(exc.value)


def test_missing_resource_path_is_input_error(order_model: dict) -> None:
    order_model["transitions"][1]["path"] = "/submit"
    with pytest.raises(ValidationError) as exc:
        StateMachineModel.model_validate(order_model)
    assert "path must contain {resource_id}" in str(exc.value)


def test_same_seed_and_requests_yield_same_schedule(order_model: dict) -> None:
    model = StateMachineModel.model_validate(order_model)

    def make() -> list[str]:
        request = RunRequest(
            model=model,
            base_url="http://127.0.0.1:1",
            scheduler="random",
            seed=42,
            max_episodes=3,
            max_steps_per_episode=8,
        )
        return [
            (
                item.episode,
                item.step,
                item.transition,
                item.rng_draws,
                item.rng_value,
            )
            for item in create_plan(request)
        ]

    assert make() == make()
    planned = make()
    random_choices = [item for item in planned if item[4] is not None]
    assert random_choices
    assert [item[3] for item in random_choices] == list(
        range(1, len(random_choices) + 1)
    )


def test_targeted_unknown_transition_is_input_error(order_model: dict) -> None:
    request = RunRequest(
        model=StateMachineModel.model_validate(order_model),
        base_url="http://127.0.0.1:1",
        target_transitions=["submit_draft"],
    )
    with pytest.raises(Exception, match="explicit illegal transitions"):
        create_plan(request)
