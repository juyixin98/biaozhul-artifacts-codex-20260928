from __future__ import annotations

import pytest

from change_detection import dp as dp_module
from change_detection.dp import solve
from change_detection.errors import InvalidParameterError, InvalidSeriesError, PruningVerificationError
from change_detection.reference import enumerate_best_segmentation


def test_zero_penalty_clean_step_has_concrete_two_segment_result() -> None:
    result = solve([0, 0, 0, 10, 10, 10], penalty=0, prune=False)

    assert result.objective == 0.0
    assert result.sse == 0.0
    assert result.total_penalty == 0.0
    assert result.breakpoints == (3,)
    assert result.segment_count == 2
    assert [(segment.start, segment.end, segment.mean, segment.cost) for segment in result.segments] == [
        (0, 3, 0.0, 0.0),
        (3, 6, 10.0, 0.0),
    ]


def test_outlier_isolation_against_reference() -> None:
    values = [0, 0, 0, 10, 0, 0, 0]
    result = solve(values, penalty=20)
    reference = enumerate_best_segmentation(values, 1, 20)

    assert result.breakpoints == (3, 4)
    assert result.segment_count == 3
    assert result.objective == pytest.approx(60.0)
    assert [segment.cost for segment in result.segments] == [0.0, 0.0, 0.0]
    assert result.objective == float(reference.objective)


def test_huge_penalty_returns_one_segment() -> None:
    values = [0, 0, 0, 10, 0, 0, 0]
    result = solve(values, penalty=100000)

    assert result.breakpoints == ()
    assert result.segment_count == 1
    assert result.objective == pytest.approx(100000.0 + 600.0 / 7.0)
    assert result.segments[0].cost == pytest.approx(600.0 / 7.0)


def test_minimum_segment_length_excludes_boundary_that_leaves_short_suffix() -> None:
    result = solve([0, 0, 10, 10], min_segment_length=2, penalty=0)

    assert result.breakpoints == (2,)
    assert result.transition_count == 3
    assert [(segment.start, segment.end) for segment in result.segments] == [(0, 2), (2, 4)]


def test_equal_objective_uses_fewer_segments_then_lexicographic_breakpoints() -> None:
    equal_count = solve([0, 10, 0], penalty=50.0 / 3.0)
    assert equal_count.breakpoints == (1,)
    assert equal_count.objective == pytest.approx(53.0)

    fewer_segments = solve([0, 10], penalty=50)
    assert fewer_segments.breakpoints == ()
    assert fewer_segments.segment_count == 1
    assert fewer_segments.objective == pytest.approx(75.0)


def test_pruning_preserves_optimum_and_records_skipped_transitions() -> None:
    values = [0] * 15 + [10] * 15
    pruned = solve(values, penalty=1000, prune=True, verify_pruning=False)
    unpruned = solve(values, penalty=1000, prune=False, verify_pruning=False)

    assert pruned.objective == unpruned.objective == pytest.approx(1750.0)
    assert pruned.breakpoints == unpruned.breakpoints == ()
    assert unpruned.transition_count == 436
    assert pruned.transition_count == 436
    assert pruned.pruned_transitions > 0


def test_pruning_verification_detects_mismatch(monkeypatch) -> None:
    original = dp_module._dynamic_programming

    def inconsistent(stats, min_segment_length, penalty, *, prune, progress, run_id=None):
        if prune:
            return original(
                stats,
                min_segment_length,
                4.0,
                prune=False,
                progress=progress,
                run_id=run_id,
            )
        return original(
            stats,
            min_segment_length,
            penalty,
            prune=False,
            progress=progress,
            run_id=run_id,
        )

    monkeypatch.setattr(dp_module, "_dynamic_programming", inconsistent)

    with pytest.raises(PruningVerificationError) as exc_info:
        solve([0, 0, 10, 10], penalty=1)
    assert exc_info.value.category == "PRUNING_MISMATCH"


def test_validation_failures_keep_specific_categories() -> None:
    with pytest.raises(InvalidSeriesError) as exc_info:
        solve([], penalty=0)
    assert exc_info.value.category == "INVALID_SERIES"

    with pytest.raises(InvalidParameterError) as exc_info:
        solve([1, 2, 3], min_segment_length=4)
    assert exc_info.value.category == "INVALID_PARAMETER"

    with pytest.raises(InvalidParameterError) as exc_info:
        solve([1, 2], penalty=-1)
    assert exc_info.value.category == "INVALID_PARAMETER"


def test_progress_events_identify_input_run_and_decision_basis() -> None:
    events = []
    solve(
        [0, 0, 10, 10],
        penalty=1,
        progress=lambda event, payload: events.append((event, payload)),
        run_id="run-log-link",
    )

    assert events[0][0] == "prefix_stats_ready"
    assert events[0][1]["run_id"] == "run-log-link"
    assert any(event == "pruning_verification_passed" for event, _ in events)
