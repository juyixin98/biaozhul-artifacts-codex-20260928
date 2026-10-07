"""Analysis/execution kernel.

Important invariant: a planned model transition is applied only after the SUT
response has been observed and accepted. Rejected operations keep the model
in its prior state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .domain import (
    Observation,
    ObservationKind,
    PlannedOperation,
    RunResult,
    StateMachineModel,
    StepRecord,
)
from .errors import ErrorCategory
from .http_adapter import HttpSutAdapter
from .model_state import ModelState

AdapterFactory = Callable[[], HttpSutAdapter]


def _failure_category(
    expected: ObservationKind,
    observation: Observation,
) -> ErrorCategory:
    if observation.category is not None:
        return observation.category
    # A successful response where a rejection was required is an acceptance
    # violation of the state contract.
    return ErrorCategory.STATE_CONFLICT if expected is ObservationKind.REJECTION else ErrorCategory.COMPUTATION_FAILURE


def _evaluate(plan: PlannedOperation, observation: Observation) -> tuple[str, str]:
    expected_observation_matches = (
        200 <= observation.status_code < 300
        if plan.expected_observation is ObservationKind.SUCCESS
        else observation.status_code == plan.expected_status
    )
    if not expected_observation_matches:
        return (
            "fail",
            (
                f"expected {plan.expected_observation} HTTP "
                f"{plan.expected_status}, observed HTTP {observation.status_code}"
            ),
        )

    if plan.expected_observation is ObservationKind.REJECTION:
        if observation.category != plan.expected_error:
            return (
                "fail",
                (
                    f"expected rejection category {plan.expected_error}, "
                    f"observed {observation.category}"
                ),
            )
        return "pass", "illegal operation rejected; model state retained"

    if observation.category is not None:
        return (
            "fail",
            f"successful operation observed error category {observation.category}",
        )
    return "pass", "successful operation observed; model advanced"


def execute_plan(
    model: StateMachineModel,
    plans: Sequence[PlannedOperation],
    adapter: HttpSutAdapter,
    synthetic_fault: str | None = None,
) -> tuple[list[StepRecord], StepRecord | None]:
    state = ModelState.initial(model)
    resource_ids: dict[int, str] = {}
    records: list[StepRecord] = []

    for plan in plans:
        if plan.episode not in resource_ids and plan.state_before == model.initial_state:
            state.reset_to_initial()
        if plan.state_before != state.state:
            reason = (
                f"plan precondition expects {plan.state_before}, model is "
                f"{state.state}"
            )
            observation = Observation(
                status_code=0,
                category=ErrorCategory.COMPUTATION_FAILURE,
                transport_error=reason,
            )
            record = StepRecord(
                plan=plan,
                observation=observation,
                state_after=state.state,
                verdict="fail",
                reason=reason,
            )
            records.append(record)
            return records, record

        resource_id = resource_ids.get(plan.episode)
        state.mark_call_dispatched()
        observation = adapter.invoke(
            plan,
            model.resource_id_placeholder,
            resource_id,
            synthetic_fault,
        )
        verdict, reason = _evaluate(plan, observation)

        if verdict == "pass":
            new_resource_id = None
            if (
                plan.state_before == model.initial_state
                and plan.expected_observation is ObservationKind.SUCCESS
                and plan.episode not in resource_ids
            ):
                body = observation.response_body
                if not isinstance(body, dict) or model.resource_id_field not in body:
                    record = StepRecord(
                        plan=plan,
                        observation=observation,
                        state_after=state.state,
                        verdict="fail",
                        reason=(
                            "create response missing "
                            f"{model.resource_id_field!r}"
                        ),
                    )
                    state.observed = True
                    records.append(record)
                    return records, record
                new_resource_id = str(body[model.resource_id_field])

            next_state = state.apply_observed(plan.transition)
            if new_resource_id is not None:
                resource_ids[plan.episode] = new_resource_id
        else:
            # A rejected call is an accepted observation and retains state.
            # Any other mismatch is a failure and is not used to update model.
            if plan.expected_observation is ObservationKind.REJECTION and (
                observation.status_code == plan.expected_status
                and observation.category == plan.expected_error
            ):
                next_state = state.apply_observed(plan.transition)
            else:
                next_state = state.state
                state.observed = True

        record = StepRecord(
            plan=plan,
            observation=observation,
            state_after=next_state,
            verdict=verdict,
            reason=reason,
        )
        records.append(record)
        if verdict == "fail":
            return records, record

    return records, None


def _sequence_preconditions(
    model: StateMachineModel,
    plans: Sequence[PlannedOperation],
) -> bool:
    if not plans:
        return False
    state = ModelState.initial(model)
    episode = plans[0].episode
    for plan in plans:
        if plan.episode != episode or plan.state_before != state.state:
            return False
        transition = state.transition(plan.transition)
        state.mark_call_dispatched()
        state.apply_observed(plan.transition)
        if transition.observation is ObservationKind.SUCCESS:
            state.state = transition.next_state
    return True


def _same_failure(
    model: StateMachineModel,
    original: StepRecord,
    candidate_plans: Sequence[PlannedOperation],
    adapter_factory: AdapterFactory,
    synthetic_fault: str | None = None,
) -> bool:
    if not _sequence_preconditions(model, candidate_plans):
        return False
    with adapter_factory() as adapter:
        records, failure = execute_plan(
            model, candidate_plans, adapter, synthetic_fault
        )
    if failure is None:
        return False
    return (
        failure.plan.transition == original.plan.transition
        and failure.observation.status_code == original.observation.status_code
        and _failure_category(failure.plan.expected_observation, failure.observation)
        == _failure_category(original.plan.expected_observation, original.observation)
        and records[-1].verdict == "fail"
    )


def reduce_failing_plan(
    model: StateMachineModel,
    plans: Sequence[PlannedOperation],
    original_failure: StepRecord,
    adapter_factory: AdapterFactory,
    synthetic_fault: str | None = None,
) -> list[PlannedOperation]:
    """Delta-debug a sequence, revalidating every retained sequence.

    A removed prefix is accepted only when the remaining operations still form
    a valid model path from the initial state and reproduce the same failure
    on a fresh SUT resource.
    """

    current = list(plans)
    if not _same_failure(
        model, original_failure, current, adapter_factory, synthetic_fault
    ):
        return current

    chunk_count = 2
    while chunk_count <= len(current):
        chunk_size = len(current) // chunk_count
        reduced = False
        for index in range(chunk_count):
            start = index * chunk_size
            end = len(current) if index == chunk_count - 1 else start + chunk_size
            if start >= end:
                continue
            candidate = current[:start] + current[end:]
            if _same_failure(
                model,
                original_failure,
                candidate,
                adapter_factory,
                synthetic_fault,
            ):
                current = candidate
                chunk_count = max(2, chunk_count - 1)
                reduced = True
                break
        if reduced:
            continue
        if chunk_count >= len(current):
            break
        chunk_count = min(len(current), chunk_count + 1)
    return current


def make_run_result(
    *,
    run_id: int | None,
    seed: int,
    scheduler_name: str,
    records: list[StepRecord],
    failure: StepRecord | None,
    replay_script: str | None = None,
    reduced_from_step_count: int | None = None,
) -> RunResult:
    if failure is None:
        return RunResult(
            run_id=run_id,
            status="passed",
            seed=seed,
            scheduler=scheduler_name,  # type: ignore[arg-type]
            steps=records,
        )
    return RunResult(
        run_id=run_id,
        status="failed",
        failure_category=_failure_category(
            failure.plan.expected_observation, failure.observation
        ),
        failure_step=failure.plan.step,
        seed=seed,
        scheduler=scheduler_name,  # type: ignore[arg-type]
        steps=records,
        replay_script=replay_script,
        reduced_from_step_count=reduced_from_step_count,
    )
