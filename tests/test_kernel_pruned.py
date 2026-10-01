import pytest

from changepoint_api.experiments.oracle import oracle_optimum
from changepoint_api.kernel import detect_mean_changepoints


def test_verifiable_pruning_preserves_exact_optimum_and_concrete_output():
    samples = [0.0] * 6 + [10.0] * 6
    exact = detect_mean_changepoints(
        samples, penalty=1.0, min_segment_length=2, algorithm="exact_dp"
    )
    pruned = detect_mean_changepoints(
        samples, penalty=1.0, min_segment_length=2, algorithm="pruned_dp"
    )

    assert pruned.change_points == [6]
    assert pruned.objective == pytest.approx(2.0)
    assert pruned.diagnostics["removed_candidate_events"] > 0
    assert pruned.diagnostics["proof_evaluations"] > 0
    assert pruned.objective == exact.objective
    assert pruned.change_points == exact.change_points
    assert [segment.cost for segment in pruned.segments] == [
        segment.cost for segment in exact.segments
    ]


@pytest.mark.parametrize(
    ("samples", "penalty", "min_length"),
    [
        ([1.0, 2.0, -3.0, 8.0, 0.0], 0.4, 1),
        ([1.0, 1.0, 7.0, 7.0, 7.0, -1.0, -1.0], 0.8, 2),
        ([0.0, 0.0, 1.0, 1.0], 0.0, 1),
        ([0.0, 0.0, 1.0, 1.0], 1.0e6, 1),
    ],
)
def test_pruned_kernel_matches_exact_and_enumeration(
    samples, penalty, min_length
):
    exact = detect_mean_changepoints(
        samples,
        penalty=penalty,
        min_segment_length=min_length,
        algorithm="exact_dp",
    )
    pruned = detect_mean_changepoints(
        samples,
        penalty=penalty,
        min_segment_length=min_length,
        algorithm="pruned_dp",
    )
    expected = oracle_optimum(samples, penalty, min_length)

    assert pruned.change_points == exact.change_points
    assert pruned.objective == pytest.approx(exact.objective, rel=1e-12)
    assert [segment.cost for segment in pruned.segments] == pytest.approx(
        [segment.cost for segment in exact.segments], rel=1e-12
    )
    assert tuple(pruned.change_points) == expected.change_points
