"""Pydantic contracts for input models, plans, observations, and runs."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import ErrorCategory, ModelValidationError

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


class ObservationKind(StrEnum):
    SUCCESS = "success"
    REJECTION = "rejection"


class SchedulerKind(StrEnum):
    TARGETED = "targeted"
    COVERAGE = "coverage"
    RANDOM = "random"


class SyntheticFault(StrEnum):
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMPUTATION_FAILURE = "computation_failure"


class TransitionSpec(BaseModel):
    """One model transition.

    Illegal transitions are explicit, so the generator does not invent calls
    that the input model did not declare.
    """

    name: str
    operation: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    body: dict[str, str] = Field(default_factory=dict)
    allowed_states: list[str] = Field(min_length=1)
    observation: ObservationKind
    expected_status: int = Field(ge=100, le=599)
    next_state: str | None = None
    expected_error: ErrorCategory | None = None

    @field_validator("name", "operation")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError("must use lowercase letters, digits, or underscore")
        return value

    @model_validator(mode="after")
    def validate_observation(self) -> "TransitionSpec":
        if self.observation is ObservationKind.SUCCESS:
            if self.next_state is None:
                raise ValueError("successful transition requires next_state")
            if self.expected_error is not None:
                raise ValueError("successful transition cannot expect an error")
            if not 200 <= self.expected_status < 300:
                raise ValueError("successful transition requires a 2xx status")
        else:
            if len(self.allowed_states) != 1:
                raise ValueError("rejection transition must name exactly one state")
            if self.next_state is not None and self.next_state != self.allowed_states[0]:
                raise ValueError("rejected operation must leave model state unchanged")
            if self.expected_error != ErrorCategory.STATE_CONFLICT:
                raise ValueError("illegal state-machine operation must be state_conflict")
            if self.expected_status != 409:
                raise ValueError("local state conflict contract uses HTTP 409")
            self.next_state = self.allowed_states[0]
        return self


class StateMachineModel(BaseModel):
    name: str
    initial_state: str
    resource_id_placeholder: str = "resource_id"
    resource_id_field: str = "id"
    states: list[str] = Field(min_length=2)
    transitions: list[TransitionSpec] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError("must use lowercase letters, digits, or underscore")
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> "StateMachineModel":
        if len(set(self.states)) != len(self.states):
            raise ValueError("states must be unique")
        if self.initial_state not in self.states:
            raise ValueError("initial_state must be listed in states")

        names: set[str] = set()
        placeholders = {self.resource_id_placeholder}
        for transition in self.transitions:
            if transition.name in names:
                raise ValueError(f"duplicate transition name: {transition.name}")
            names.add(transition.name)
            unknown = set(transition.allowed_states) - set(self.states)
            if unknown:
                raise ValueError(f"unknown state in {transition.name}: {sorted(unknown)}")
            if transition.next_state not in self.states:
                raise ValueError(f"unknown next_state in {transition.name}")
            needs_resource_id = transition.allowed_states != [self.initial_state]
            if needs_resource_id and (
                "{" + self.resource_id_placeholder + "}" not in transition.path
            ):
                raise ValueError(
                    f"{transition.name} path must contain "
                    f"{{{self.resource_id_placeholder}}}"
                )
        return self

    def applicable(self, state: str) -> list[TransitionSpec]:
        return [item for item in self.transitions if state in item.allowed_states]

    def legal(self, state: str) -> list[TransitionSpec]:
        return [
            item
            for item in self.applicable(state)
            if item.observation is ObservationKind.SUCCESS
        ]

    def illegal(self, state: str) -> list[TransitionSpec]:
        return [
            item
            for item in self.applicable(state)
            if item.observation is ObservationKind.REJECTION
        ]


class RunRequest(BaseModel):
    model: StateMachineModel
    base_url: str = Field(min_length=1)
    scheduler: SchedulerKind = SchedulerKind.TARGETED
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_episodes: int = Field(default=3, ge=1, le=100)
    max_steps_per_episode: int = Field(default=12, ge=1, le=200)
    synthetic_fault: SyntheticFault | None = None
    target_transitions: list[str] | None = Field(default=None, min_length=1)

    @field_validator("target_transitions")
    @classmethod
    def validate_targets(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("target_transitions must be unique")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must be http(s) URL")
        return value.rstrip("/")


class PlannedOperation(BaseModel):
    episode: int
    step: int
    transition: str
    operation: str
    method: str
    path: str
    body: dict[str, str]
    state_before: str
    expected_observation: ObservationKind
    expected_status: int
    expected_next_state: str
    expected_error: ErrorCategory | None = None
    choice_reason: str
    choice_index: int
    rng_draws: int = 0
    rng_value: int | None = None


class Observation(BaseModel):
    status_code: int
    category: ErrorCategory | None = None
    response_body: object | None = None
    transport_error: str | None = None


class StepRecord(BaseModel):
    plan: PlannedOperation
    observation: Observation
    state_after: str
    verdict: Literal["pass", "fail"]
    reason: str


class RunResult(BaseModel):
    run_id: int | None = None
    status: Literal["passed", "failed", "input_error", "computation_failure"]
    failure_category: ErrorCategory | None = None
    failure_step: int | None = None
    seed: int
    scheduler: SchedulerKind
    steps: list[StepRecord]
    replay_script: str | None = None
    reduced_from_step_count: int | None = None


def validate_model_or_raise(model: StateMachineModel) -> None:
    """Translate Pydantic/graph problems into the shared input category."""

    try:
        StateMachineModel.model_validate(model.model_dump())
    except ValueError as exc:
        raise ModelValidationError(str(exc)) from exc
