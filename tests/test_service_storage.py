from __future__ import annotations

import pytest

from app.errors import (
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
    ResourceExhaustedError,
    StateConflictError,
)
from app.service import BetaBinomialService
from app.storage import SQLiteStore
from tests.replay_logger import ReplayLogger


@pytest.fixture
def service() -> BetaBinomialService:
    return BetaBinomialService(SQLiteStore(":memory:"))


def test_batched_updates_equal_one_combined_posterior(
    service: BetaBinomialService, replay_logger: ReplayLogger
) -> None:
    service.create_run("split", 2, 3)
    service.submit_batch("split", "A", 3, 1)
    service.submit_batch("split", "B", 2, 4)
    split = service.posterior_summary("split")

    service.create_run("combined", 2, 3)
    service.submit_batch("combined", "ALL", 5, 5)
    combined = service.posterior_summary("combined")

    replay_logger.state(
        "posterior_equivalence",
        {
            "split": [split.posterior_alpha, split.posterior_beta],
            "combined": [combined.posterior_alpha, combined.posterior_beta],
        },
        "Prior (2,3) plus aggregate (5,5) must be Beta(7,8).",
    )
    assert (
        split.posterior_alpha,
        split.posterior_beta,
        split.total_successes,
        split.total_failures,
    ) == (7, 8, 5, 5)
    assert (split.posterior_alpha, split.posterior_beta) == (
        combined.posterior_alpha,
        combined.posterior_beta,
    )


def test_batch_correction_withdraws_counts_and_versions_remain(
    service: BetaBinomialService, replay_logger: ReplayLogger
) -> None:
    service.create_run("correct", 1, 1)
    service.submit_batch("correct", "A", 3, 1, reason="initial")
    before = service.posterior_summary("correct")
    revised = service.revise_batch(
        "correct", "A", 1, 1, reason="two successes were double counted"
    )
    after = service.posterior_summary("correct")
    versions = service.list_versions("correct")
    batch_versions = service.store.list_batch_versions("correct")

    replay_logger.state(
        "revision",
        {
            "before": [before.posterior_alpha, before.posterior_beta],
            "after": [after.posterior_alpha, after.posterior_beta],
            "batch_statuses": [row["status"] for row in batch_versions],
        },
        "The old (3,1) row remains superseded and current totals use (1,1).",
    )
    assert before.total_successes == 3
    assert revised.event_type == "batch_revised"
    assert revised.version == 2
    assert (after.total_successes, after.total_failures) == (1, 1)
    assert (after.posterior_alpha, after.posterior_beta) == (2, 2)
    assert [row["status"] for row in batch_versions] == ["superseded", "active"]
    assert [version.event_type for version in versions] == [
        "created",
        "batch_submitted",
        "batch_revised",
    ]


def test_batch_revision_can_recover_all_success_to_mixed_counts(
    service: BetaBinomialService, replay_logger: ReplayLogger
) -> None:
    service.create_run("recover", 1, 1)
    service.submit_batch("recover", "bad", 10, 0)
    all_success = service.predict("recover", 1)
    service.revise_batch("recover", "bad", 6, 4, reason="corrected labels")
    recovered = service.predict("recover", 1)

    replay_logger.state(
        "recovery",
        {
            "all_success_probabilities": [
                point.probability for point in all_success.points
            ],
            "recovered_posterior": [
                recovered.posterior_alpha,
                recovered.posterior_beta,
            ],
        },
        "Revising 10/0 to 6/4 must move posterior from Beta(11,1) to Beta(7,5).",
    )
    assert (recovered.posterior_alpha, recovered.posterior_beta) == (7, 5)
    assert recovered.probability_sum == pytest.approx(1.0, abs=1e-15)


def test_state_conflicts_are_specific(service: BetaBinomialService) -> None:
    service.create_run("run", 1, 1)
    service.submit_batch("run", "A", 1, 1)

    with pytest.raises(StateConflictError, match="already exists"):
        service.submit_batch("run", "A", 2, 2)
    with pytest.raises(StateConflictError, match="already exists"):
        service.create_run("run", 1, 1)
    with pytest.raises(NotFoundError, match="never been submitted"):
        service.revise_batch("run", "missing", 1, 1)
    with pytest.raises(StateConflictError, match="Run does not exist"):
        service.submit_batch("missing", "A", 1, 1)


def test_input_error_categories_are_distinct(service: BetaBinomialService) -> None:
    service.create_run("run", 1, 1)
    with pytest.raises(InvalidInputError, match="non-negative"):
        service.submit_batch("run", "B", -1, 0)
    with pytest.raises(InvalidInputError, match="strictly positive"):
        service.create_run("bad", 0, 1)


def test_prediction_size_is_resource_bound(service: BetaBinomialService) -> None:
    service.create_run("run", 1, 1)
    with pytest.raises(PayloadTooLargeError, match="too many prediction"):
        service.predict("run", 10_001)
    with pytest.raises(ResourceExhaustedError, match="SQLite limit"):
        service.submit_batch("run", "huge", 2**63, 0)
