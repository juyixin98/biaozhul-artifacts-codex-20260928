from __future__ import annotations

from mbt_server.core.domain import (
    Observation,
    ObservationKind,
    PlannedOperation,
    RunRequest,
    StateMachineModel,
)
from mbt_server.core.engine import run_model_test
from mbt_server.core.errors import ErrorCategory
from mbt_server.core.storage import EvidenceStore


CORE_TARGETS = [
    "reject_close_closed",
    "reject_close_draft",
    "reject_submit_closed",
    "reject_cancel_closed",
]


def _request(order_model: dict, base_url: str) -> RunRequest:
    return RunRequest(
        model=StateMachineModel.model_validate(order_model),
        base_url=base_url,
        scheduler="targeted",
        seed=342,
        target_transitions=CORE_TARGETS,
    )


def test_reference_service_passes_required_illegal_sequences(
    order_model: dict,
    reference_url: str,
    tmp_path,
) -> None:
    with EvidenceStore(tmp_path / "reference.sqlite3") as store:
        result = run_model_test(_request(order_model, reference_url), store)

    assert result.status == "passed"
    assert result.failure_category is None
    assert len(result.steps) == 14
    assert [step.plan.transition for step in result.steps] == [
        "create_order",
        "submit_draft",
        "close_submitted",
        "reject_close_closed",
        "create_order",
        "reject_close_draft",
        "create_order",
        "submit_draft",
        "close_submitted",
        "reject_submit_closed",
        "create_order",
        "submit_draft",
        "close_submitted",
        "reject_cancel_closed",
    ]
    assert all(step.verdict == "pass" for step in result.steps)
    assert [step.state_after for step in result.steps] == [
        "draft",
        "submitted",
        "closed",
        "closed",
        "draft",
        "draft",
        "draft",
        "submitted",
        "closed",
        "closed",
        "draft",
        "submitted",
        "closed",
        "closed",
    ]


def test_buggy_service_failure_has_category_states_reason_and_replay(
    order_model: dict,
    buggy_url: str,
    tmp_path,
) -> None:
    with EvidenceStore(tmp_path / "buggy.sqlite3") as store:
        result = run_model_test(_request(order_model, buggy_url), store)
        run_id = result.run_id
        persisted = store.get_run(run_id)
        replay = store.get_replay_script(run_id)

    assert result.status == "failed"
    assert result.failure_category is ErrorCategory.STATE_CONFLICT
    failure = result.steps[3]
    assert failure.plan.transition == "reject_close_closed"
    assert failure.observation.status_code == 200
    assert failure.state_after == "closed"
    assert "expected rejection HTTP 409, observed HTTP 200" in failure.reason
    assert result.reduced_from_step_count == 4

    assert persisted is not None
    assert persisted["failure_category"] == "state_conflict"
    assert persisted["seed"] == 342
    failed_step = persisted["steps"][3]
    assert failed_step["state_before"] == "closed"
    assert failed_step["observed_status"] == 200
    assert failed_step["choice_reason"] == "explicit illegal target"

    assert replay is not None
    assert "urllib.request" in replay
    assert "/orders" in replay
    assert "/close" in replay


def test_reduced_replay_is_minimal_and_revalidates_fresh_resource(
    order_model: dict,
    buggy_url: str,
    tmp_path,
) -> None:
    request = _request(order_model, buggy_url)
    request.target_transitions = ["reject_close_draft"]
    with EvidenceStore(tmp_path / "reduced.sqlite3") as store:
        result = run_model_test(request, store)
        replay = store.get_replay_script(result.run_id)

    assert result.status == "failed"
    assert result.steps[-1].plan.transition == "reject_close_draft"
    assert replay is not None
    # The create step is required; the submit prefix must have been removed
    # only after a fresh-service replay still proved the draft-close failure.
    assert replay.count('"path": "/orders"') == 1
    assert replay.count('"path": "/orders/{resource_id}/close"') == 1
    assert "/submit" not in replay


def test_synthetic_faults_are_reported_with_distinct_categories(
    order_model: dict,
    reference_url: str,
    tmp_path,
) -> None:
    model = StateMachineModel.model_validate(order_model)
    request = RunRequest(
        model=model,
        base_url=reference_url,
        scheduler="targeted",
        target_transitions=["reject_close_draft"],
        synthetic_fault="resource_exhausted",
    )
    with EvidenceStore(tmp_path / "faults.sqlite3") as store:
        result = run_model_test(request, store)

    assert result.status == "failed"
    assert result.failure_category is ErrorCategory.RESOURCE_EXHAUSTED
    assert result.steps[0].observation.status_code == 507


def test_http_input_error_and_transport_failure_have_distinct_categories(
    order_model: dict,
    reference_url: str,
    tmp_path,
) -> None:
    model = StateMachineModel.model_validate(order_model)
    valid_body = model.transitions[0].body
    model.transitions[0].body = {"id": ""}
    request = RunRequest(
        model=model,
        base_url=reference_url,
        scheduler="targeted",
        target_transitions=["reject_close_draft"],
    )
    with EvidenceStore(tmp_path / "http_categories.sqlite3") as store:
        result = run_model_test(request, store)
    assert result.failure_category is ErrorCategory.INPUT_ERROR
    assert result.steps[0].observation.status_code == 422

    model.transitions[0].body = valid_body
    request.base_url = "http://127.0.0.1:1"
    with EvidenceStore(tmp_path / "transport.sqlite3") as store:
        result = run_model_test(request, store)
    assert result.failure_category is ErrorCategory.COMPUTATION_FAILURE
    assert result.steps[0].observation.status_code == 0
    assert result.steps[0].observation.transport_error is not None


def test_incomplete_success_observation_does_not_advance_model(
    order_model: dict,
) -> None:
    model = StateMachineModel.model_validate(order_model)
    transition = next(
        item for item in model.transitions if item.name == "create_order"
    )
    plan = PlannedOperation(
        episode=0,
        step=1,
        transition=transition.name,
        operation=transition.operation,
        method=transition.method,
        path=transition.path,
        body={},
        state_before="none",
        expected_observation=ObservationKind.SUCCESS,
        expected_status=201,
        expected_next_state="draft",
        choice_reason="unit edge case",
        choice_index=0,
    )

    class MissingIdAdapter:
        def invoke(self, *_args, **_kwargs) -> Observation:
            return Observation(status_code=201, response_body={"state": "draft"})

    from mbt_server.core.runner import execute_plan

    records, failure = execute_plan(model, [plan], MissingIdAdapter())  # type: ignore[arg-type]
    assert failure is not None
    assert failure.state_after == "none"
    assert failure.verdict == "fail"
    assert "missing 'id'" in failure.reason
