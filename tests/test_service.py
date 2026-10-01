from __future__ import annotations

import pytest

from app.errors import BatchNotFoundError, ResourceExhaustedError, StateConflictError
from app.services.beta_service import BetaBinomialService
from experiments.oracle import beta_posterior


def test_batches_split_then_merge_match_one_shot_posterior(service: BetaBinomialService, test_logger):
    split = service.create_model(1.0, 1.0)
    b1 = service.add_batch(split["id"], 3, 2, reason="first split")
    b2 = service.add_batch(split["id"], 5, 1, reason="second split")
    split_state = service.current_state(split["id"])

    merged = service.create_model(1.0, 1.0)
    service.add_batch(merged["id"], 8, 3, reason="one combined batch")
    merged_state = service.current_state(merged["id"])

    test_logger.note(
        "split_matches_merged",
        "compare_state",
        {
            "batch_ids": [b1["id"], b2["id"]],
            "split_totals": [split_state["success_count"], split_state["failure_count"]],
            "merged_totals": [merged_state["success_count"], merged_state["failure_count"]],
            "split_version": split_state["version"],
            "merged_version": merged_state["version"],
        },
        "Conjugate updates depend only on sufficient totals, so version histories may differ while posteriors match.",
    )

    assert (split_state["success_count"], split_state["failure_count"]) == (8, 3)
    assert (
        split_state["success_count"],
        split_state["failure_count"],
    ) == (
        merged_state["success_count"],
        merged_state["failure_count"],
    )
    split_summary = service.posterior_summary(split["id"])
    merged_summary = service.posterior_summary(merged["id"])
    assert split_summary["posterior_alpha"] == merged_summary["posterior_alpha"]
    assert split_summary["posterior_beta"] == merged_summary["posterior_beta"]

    oracle = beta_posterior(1.0, 1.0, 8, 3)
    assert split_summary["posterior_alpha"] == pytest.approx(oracle.alpha)
    assert split_summary["posterior_beta"] == pytest.approx(oracle.beta)


def test_batch_correction_withdraws_then_adds_and_recovery_restores(
    service: BetaBinomialService, test_logger
):
    model = service.create_model(2.0, 3.0)
    wrong = service.add_batch(model["id"], 10, 5, reason="misread form")
    service.add_batch(model["id"], 1, 1, reason="unaffected batch")
    before = service.current_state(model["id"])

    corrected = service.revise_batch(
        model["id"], wrong["id"], 4, 7, reason="correct first form"
    )
    after_revision = service.current_state(model["id"])
    events = service.list_events(model["id"])
    withdrawal = events[-2]
    addition = events[-1]

    assert withdrawal["event_type"] == "BATCH_REVISION_WITHDREW_ORIGINAL"
    assert (withdrawal["delta_success"], withdrawal["delta_failure"]) == (-10, -5)
    assert addition["event_type"] == "BATCH_REVISION_ADDED_REPLACEMENT"
    assert (addition["delta_success"], addition["delta_failure"]) == (4, 7)
    assert (after_revision["success_count"], after_revision["failure_count"]) == (5, 8)

    recovered = service.restore_batch(model["id"], wrong["id"], reason="original was right")
    after_recovery = service.current_state(model["id"])
    events = service.list_events(model["id"])
    assert events[-2]["event_type"] == "BATCH_RECOVERY_WITHDREW_REPLACEMENT"
    assert (events[-2]["delta_success"], events[-2]["delta_failure"]) == (-4, -7)
    assert events[-1]["event_type"] == "BATCH_RECOVERY_RESTORED_ORIGINAL"
    assert (events[-1]["delta_success"], events[-1]["delta_failure"]) == (10, 5)
    assert (after_recovery["success_count"], after_recovery["failure_count"]) == (11, 6)

    test_logger.note(
        "revision_and_recovery",
        "final_state",
        {
            "wrong_batch_id": wrong["id"],
            "replacement_batch_id": corrected["id"],
            "recovered_batch_id": recovered["id"],
            "before_totals": [before["success_count"], before["failure_count"]],
            "after_revision": [after_revision["success_count"], after_revision["failure_count"]],
            "after_recovery": [after_recovery["success_count"], after_recovery["failure_count"]],
            "event_types": [event["event_type"] for event in events],
        },
        "Revision totals changed (11,6)->(5,8); recovery reversed the correction and returned to (11,6).",
    )


def test_revert_and_restore_standalone_batch(service: BetaBinomialService):
    model = service.create_model(1.0, 1.0)
    batch = service.add_batch(model["id"], 2, 8)
    service.revert_batch(model["id"], batch["id"])
    assert (service.current_state(model["id"])["success_count"], service.current_state(model["id"])["failure_count"]) == (0, 0)

    service.restore_batch(model["id"], batch["id"])
    state = service.current_state(model["id"])
    assert (state["success_count"], state["failure_count"]) == (2, 8)


def test_state_conflict_revision_and_restore_are_distinct(service: BetaBinomialService):
    model = service.create_model(1.0, 1.0)
    batch = service.add_batch(model["id"], 1, 1)
    service.revise_batch(model["id"], batch["id"], 2, 2)

    with pytest.raises(StateConflictError) as revise_exc:
        service.revise_batch(model["id"], batch["id"], 3, 3)
    assert revise_exc.value.category == "state_conflict"

    with pytest.raises(BatchNotFoundError) as restore_exc:
        service.restore_batch(model["id"], batch["id"] + "missing")
    assert restore_exc.value.category == "not_found"


def test_aggregate_capacity_is_resource_exhausted(service: BetaBinomialService):
    model = service.create_model(1.0, 1.0)
    service.add_batch(model["id"], service.settings.max_observation_count - 1, 0)
    with pytest.raises(ResourceExhaustedError) as exc_info:
        service.add_batch(model["id"], 2, 0)
    assert exc_info.value.category == "resource_exhausted"
    assert exc_info.value.details["future_success"] == service.settings.max_observation_count + 1
