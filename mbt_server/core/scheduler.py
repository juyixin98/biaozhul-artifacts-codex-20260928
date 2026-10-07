"""Deterministic operation-sequence schedulers.

The scheduler only plans model operations. It never invokes the SUT and never
mutates model state from an observation; those responsibilities belong to
the runner and ModelState.
"""

from __future__ import annotations

import random
from collections import deque

from .domain import (
    ObservationKind,
    PlannedOperation,
    RunRequest,
    SchedulerKind,
    StateMachineModel,
    TransitionSpec,
)
from .errors import ModelValidationError


def _plan(
    model: StateMachineModel,
    episode: int,
    step: int,
    transition: TransitionSpec,
    state_before: str,
    reason: str,
    choice_index: int,
    rng_draws: int,
    rng_value: int | None,
) -> PlannedOperation:
    assert transition.next_state is not None
    return PlannedOperation(
        episode=episode,
        step=step,
        transition=transition.name,
        operation=transition.operation,
        method=transition.method,
        path=transition.path,
        body=dict(transition.body),
        state_before=state_before,
        expected_observation=transition.observation,
        expected_status=transition.expected_status,
        expected_next_state=transition.next_state,
        expected_error=transition.expected_error,
        choice_reason=reason,
        choice_index=choice_index,
        rng_draws=rng_draws,
        rng_value=rng_value,
    )


def _success_path_to(model: StateMachineModel, target_state: str) -> list[str]:
    """Breadth-first shortest successful path from initial to target state."""

    queue: deque[tuple[str, list[str]]] = deque(
        [(model.initial_state, [])]
    )
    visited = {model.initial_state}
    while queue:
        state, path = queue.popleft()
        if state == target_state:
            return path
        for transition in model.legal(state):
            assert transition.next_state is not None
            if transition.next_state not in visited:
                visited.add(transition.next_state)
                queue.append((transition.next_state, [*path, transition.name]))
    raise ModelValidationError(f"state is not reachable: {target_state}")


def _targeted(
    model: StateMachineModel,
    targets: list[str] | None = None,
) -> list[PlannedOperation]:
    """One fresh-resource episode for every explicit illegal transition."""

    illegal = [
        transition
        for transition in model.transitions
        if transition.observation is ObservationKind.REJECTION
    ]
    if targets is not None:
        known = {transition.name for transition in illegal}
        unknown = sorted(set(targets) - known)
        if unknown:
            raise ModelValidationError(
                f"target transitions are not explicit illegal transitions: {unknown}"
            )
        target_order = {name: index for index, name in enumerate(targets)}
        selected = [item for item in illegal if item.name in target_order]
        illegal = sorted(selected, key=lambda item: target_order[item.name])
    plans: list[PlannedOperation] = []
    for episode, illegal_transition in enumerate(illegal):
        state = model.initial_state
        names = _success_path_to(model, illegal_transition.allowed_states[0])
        for step, name in enumerate(names, start=1):
            transition = next(item for item in model.transitions if item.name == name)
            plans.append(
                _plan(
                    model,
                    episode,
                    step,
                    transition,
                    state,
                    "targeted legal prefix",
                    0,
                    0,
                    None,
                )
            )
            state = transition.next_state  # predicted BFS state, not an observation
        plans.append(
            _plan(
                model,
                episode,
                len(names) + 1,
                illegal_transition,
                state,
                "explicit illegal target",
                0,
                0,
                None,
            )
        )
    return plans


def _coverage(model: StateMachineModel, max_episodes: int) -> list[PlannedOperation]:
    """BFS legal prefix plus one illegal operation for each state/operation pair."""

    pairs: list[tuple[str, TransitionSpec]] = []
    for state in model.states:
        for transition in model.illegal(state):
            pairs.append((state, transition))

    prefix_cache: dict[str, list[str]] = {}
    plans: list[PlannedOperation] = []
    for episode, (target_state, illegal_transition) in enumerate(
        pairs[:max_episodes]
    ):
        if target_state not in prefix_cache:
            prefix_cache[target_state] = _success_path_to(model, target_state)
        state = model.initial_state
        for step, name in enumerate(prefix_cache[target_state], start=1):
            transition = next(item for item in model.transitions if item.name == name)
            plans.append(
                _plan(
                    model,
                    episode,
                    step,
                    transition,
                    state,
                    "coverage legal prefix",
                    0,
                    0,
                    None,
                )
            )
            state = transition.next_state
        plans.append(
            _plan(
                model,
                episode,
                len(prefix_cache[target_state]) + 1,
                illegal_transition,
                state,
                "coverage illegal pair",
                0,
                0,
                None,
            )
        )
    return plans


def _random(
    model: StateMachineModel,
    seed: int,
    max_episodes: int,
    max_steps: int,
) -> list[PlannedOperation]:
    rng = random.Random(seed)
    plans: list[PlannedOperation] = []
    rng_draws = 0

    for episode in range(max_episodes):
        state = model.initial_state
        for step in range(1, max_steps + 1):
            candidates = model.applicable(state)
            if not candidates:
                break
            if len(candidates) == 1:
                index = 0
                rng_value = None
                reason = "single applicable transition"
            else:
                index = rng.randrange(len(candidates))
                rng_value = index
                rng_draws += 1
                reason = "seeded random applicable transition"
            transition = candidates[index]
            plans.append(
                _plan(
                    model,
                    episode,
                    step,
                    transition,
                    state,
                    reason,
                    index,
                    rng_draws,
                    rng_value,
                )
            )
            assert transition.next_state is not None
            state = transition.next_state

            # A rejected call keeps the resource; terminal success ends episode.
            if transition.observation is ObservationKind.SUCCESS and not model.applicable(
                transition.next_state
            ):
                break
    return plans


def create_plan(request: RunRequest) -> list[PlannedOperation]:
    model = request.model
    if request.scheduler is SchedulerKind.TARGETED:
        plans = _targeted(model, request.target_transitions)
    elif request.scheduler is SchedulerKind.COVERAGE:
        plans = _coverage(model, request.max_episodes)
    else:
        plans = _random(
            model,
            request.seed,
            request.max_episodes,
            request.max_steps_per_episode,
        )
    if not plans:
        raise ModelValidationError("scheduler produced no operations")
    return plans
